#!/usr/bin/env python3
"""
newsletter_crawler.py - Complete newsletter pipeline automation

This script:
1. Determines which sources to crawl based on today's date and their Check Frequency
2. Queries Notion Source Registry for active sources
3. Crawls each source using Crawl4AI
4. Deduplicates against Newsletter Pipeline (skips existing URLs)
5. Creates new Pipeline entries
6. Updates Source Registry with crawl status

Environment variables required:
- NOTION_API_KEY: Your Notion integration token

Usage:
    python newsletter_crawler.py              # Normal run (respects frequencies)
    python newsletter_crawler.py --force-all  # Crawl ALL active sources regardless of frequency
    python newsletter_crawler.py --dry-run    # Show what would be crawled without doing it
"""

import asyncio
import argparse
import os
import sys
import re
import time
import logging
from datetime import datetime, date
from typing import Optional
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Notion Database IDs
SOURCE_REGISTRY_DB = "43d593469bb8458c96ce927600514907"
NEWSLETTER_PIPELINE_DB = "2efabbce69a280409309d052751eec14"

# Data Source IDs (for the new Notion API 2025-09-03)
# These are the collection IDs from your Notion databases
SOURCE_REGISTRY_DS = "902cd405-8998-4dc8-8810-ae8f54e3f61a"
NEWSLETTER_PIPELINE_DS = "2efabbce-69a2-8016-b362-000bfe5c9a11"

# Category to Topic mapping
CATEGORY_TO_TOPIC = {
    "Church Data/Research": "Church Tech",
    "Competitor": "Church Tech",
    "Church Tech": "Church Tech",
    "Church Leadership": "Church Tech",
    "Church Conferences": "Church Tech",
    "Product Management": "Product Management",
    "AI/Practical": "AI/ML",
    "Software Trends": "Software Trends",
}

# Defaults
DEFAULT_MAX_PAGES = 5
DEFAULT_MAX_DEPTH = 2
MAX_CONSECUTIVE_FAILURES = 3

# ============================================================================
# NOTION CLIENT
# ============================================================================

try:
    from notion_client import Client
except ImportError:
    logger.error("notion-client not installed. Run: pip install notion-client")
    sys.exit(1)

def get_notion_client() -> Client:
    """Initialize Notion client with API key from environment."""
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        logger.error("NOTION_API_KEY environment variable not set")
        sys.exit(1)
    return Client(auth=api_key)


def query_data_source(notion: Client, data_source_id: str, filter_query=None, start_cursor=None, page_size=None):
    """
    Query a Notion data source using the new API (2025-09-03).
    Falls back to databases.query for older SDK versions.
    """
    kwargs = {"data_source_id": data_source_id}
    if filter_query is not None:
        kwargs["filter"] = filter_query
    if start_cursor is not None:
        kwargs["start_cursor"] = start_cursor
    if page_size is not None:
        kwargs["page_size"] = page_size
    
    # Try new API first (data_sources.query)
    if hasattr(notion, "data_sources") and hasattr(notion.data_sources, "query"):
        return notion.data_sources.query(**kwargs)
    
    # Fall back to old API (databases.query) for older SDK versions
    if hasattr(notion, "databases") and hasattr(notion.databases, "query"):
        # Old API uses database_id instead of data_source_id
        kwargs["database_id"] = data_source_id
        del kwargs["data_source_id"]
        return notion.databases.query(**kwargs)
    
    raise RuntimeError("Notion SDK does not support data_sources.query or databases.query")

# ============================================================================
# FREQUENCY LOGIC
# ============================================================================

def get_frequencies_for_today() -> list[str]:
    """
    Determine which check frequencies should run today.
    
    - Daily: every day
    - Weekly: Mondays (weekday 0)
    - Monthly: 1st of month
    - Quarterly: 1st of Jan, Apr, Jul, Oct
    """
    today = date.today()
    frequencies = ["Daily"]  # Always include daily
    
    # Weekly on Mondays
    if today.weekday() == 0:
        frequencies.append("Weekly")
    
    # Monthly on 1st
    if today.day == 1:
        frequencies.append("Monthly")
    
    # Quarterly on 1st of Jan, Apr, Jul, Oct
    if today.day == 1 and today.month in [1, 4, 7, 10]:
        frequencies.append("Quarterly")
    
    return frequencies

# ============================================================================
# NOTION QUERIES
# ============================================================================

def query_active_sources(notion: Client, frequencies: list[str]) -> list[dict]:
    """
    Query Source Registry for active sources matching given frequencies.
    """
    # Build OR filter for frequencies
    frequency_filters = [
        {"property": "Check Frequency", "select": {"equals": freq}}
        for freq in frequencies
    ]
    
    filter_query = {
        "and": [
            {"property": "Status", "select": {"equals": "Active"}},
            {"or": frequency_filters} if len(frequency_filters) > 1 else frequency_filters[0]
        ]
    }
    
    sources = []
    has_more = True
    start_cursor = None
    
    while has_more:
        response = query_data_source(
            notion,
            SOURCE_REGISTRY_DS,
            filter_query=filter_query,
            start_cursor=start_cursor
        )
        
        for page in response["results"]:
            props = page["properties"]
            
            # Extract values safely
            name = ""
            if props.get("Name", {}).get("title"):
                name = props["Name"]["title"][0]["plain_text"] if props["Name"]["title"] else ""
            
            url = props.get("URL", {}).get("url", "")
            
            category = ""
            if props.get("Category", {}).get("select"):
                category = props["Category"]["select"]["name"]
            
            crawl_pattern = ""
            if props.get("Crawl Pattern", {}).get("rich_text"):
                crawl_pattern = props["Crawl Pattern"]["rich_text"][0]["plain_text"] if props["Crawl Pattern"]["rich_text"] else ""
            
            max_pages = props.get("Max Pages", {}).get("number") or DEFAULT_MAX_PAGES
            max_depth = props.get("Max Depth", {}).get("number") or DEFAULT_MAX_DEPTH
            consecutive_failures = props.get("Consecutive Failures", {}).get("number") or 0
            
            check_frequency = ""
            if props.get("Check Frequency", {}).get("select"):
                check_frequency = props["Check Frequency"]["select"]["name"]
            
            sources.append({
                "page_id": page["id"],
                "name": name,
                "url": url,
                "category": category,
                "crawl_pattern": crawl_pattern,
                "max_pages": int(max_pages),
                "max_depth": int(max_depth),
                "consecutive_failures": int(consecutive_failures),
                "check_frequency": check_frequency,
            })
        
        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")
    
    return sources


def url_exists_in_pipeline(notion: Client, url: str) -> bool:
    """Check if a URL already exists in the Newsletter Pipeline."""
    response = query_data_source(
        notion,
        NEWSLETTER_PIPELINE_DS,
        filter_query={"property": "URL", "url": {"equals": url}},
        page_size=1
    )
    return len(response["results"]) > 0


def create_pipeline_entry(
    notion: Client,
    title: str,
    url: str,
    summary: str,
    topic: str,
    source_name: str
) -> str:
    """Create a new entry in the Newsletter Pipeline. Returns page ID."""
    
    # Build topic multi-select
    topic_value = [{"name": topic}] if topic else []
    
    # Truncate summary if too long (Notion limit is 2000 chars for rich_text)
    if len(summary) > 1900:
        summary = summary[:1900] + "..."
    
    properties = {
        "Title": {"title": [{"text": {"content": title[:2000]}}]},  # Notion title limit
        "URL": {"url": url},
        "Source": {"select": {"name": "Web Crawl"}},
        "Status": {"select": {"name": "Unreviewed"}},
        "Date Found": {"date": {"start": date.today().isoformat()}},
    }
    
    if topic_value:
        properties["Topic"] = {"multi_select": topic_value}
    
    if summary:
        properties["Notes"] = {"rich_text": [{"text": {"content": summary}}]}
    
    response = notion.pages.create(
        parent={"database_id": NEWSLETTER_PIPELINE_DB},
        properties=properties
    )
    
    return response["id"]


def update_source_status(
    notion: Client,
    page_id: str,
    success: bool,
    articles_found: int,
    pages_crawled: int,
    duration: float,
    error: Optional[str],
    current_failures: int
) -> None:
    """Update Source Registry entry with crawl results."""
    
    today_iso = date.today().isoformat()
    
    properties = {
        "Last Reviewed": {"date": {"start": today_iso}},
    }
    
    if success and articles_found > 0:
        # Success with content
        properties["Last Success"] = {"date": {"start": today_iso}}
        properties["Consecutive Failures"] = {"number": 0}
        properties["Crawl Notes"] = {
            "rich_text": [{"text": {"content": f"Found {articles_found} articles, crawled {pages_crawled} pages in {duration:.1f}s"}}]
        }
    elif success and articles_found == 0:
        # Success but no new content (not a failure)
        properties["Crawl Notes"] = {
            "rich_text": [{"text": {"content": f"No new articles found. Crawled {pages_crawled} pages in {duration:.1f}s"}}]
        }
    else:
        # Failure
        new_failures = current_failures + 1
        properties["Consecutive Failures"] = {"number": new_failures}
        
        error_msg = error or "Unknown error"
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        
        if new_failures >= MAX_CONSECUTIVE_FAILURES:
            properties["Status"] = {"select": {"name": "Needs Review"}}
            properties["Crawl Notes"] = {
                "rich_text": [{"text": {"content": f"Auto-paused after {new_failures} failures: {error_msg}"}}]
            }
        else:
            properties["Crawl Notes"] = {
                "rich_text": [{"text": {"content": f"Failure {new_failures}/{MAX_CONSECUTIVE_FAILURES}: {error_msg}"}}]
            }
    
    notion.pages.update(page_id=page_id, properties=properties)

# ============================================================================
# CRAWL4AI CRAWLER
# ============================================================================

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("Crawl4AI not installed. Run: pip install crawl4ai && crawl4ai-setup")


def extract_title_from_markdown(markdown: str, url: str) -> str:
    """Extract title from markdown content or fall back to URL."""
    if not markdown:
        return url_to_title(url)
    
    # Try to find first H1
    h1_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Try first H2
    h2_match = re.search(r'^##\s+(.+)$', markdown, re.MULTILINE)
    if h2_match:
        return h2_match.group(1).strip()
    
    # Fall back to first non-empty line
    for line in markdown.split('\n'):
        line = line.strip()
        if line and len(line) < 200 and not line.startswith('http') and not line.startswith('['):
            return line
    
    return url_to_title(url)


def url_to_title(url: str) -> str:
    """Convert URL path to readable title."""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return parsed.netloc
    
    segments = path.split('/')
    title = segments[-1] if segments else path
    
    title = title.replace('-', ' ').replace('_', ' ')
    title = re.sub(r'\.(html?|php|aspx?)$', '', title, flags=re.IGNORECASE)
    title = title.title()
    
    return title if title else parsed.netloc


def extract_date_from_content(markdown: str, url: str) -> Optional[str]:
    """Try to extract publication date from content or URL."""
    date_patterns = [
        r'(?:Published|Posted|Date|Updated)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        r'(?:Published|Posted|Date|Updated)[\s:]*(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\w+\s+\d{1,2},?\s+\d{4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, markdown, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%B %d %Y']:
                try:
                    dt = datetime.strptime(date_str.replace(',', ''), fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    
    # Try URL date pattern
    url_date_match = re.search(r'/(\d{4})[/\-](\d{2})[/\-](\d{2})/', url)
    if url_date_match:
        return f"{url_date_match.group(1)}-{url_date_match.group(2)}-{url_date_match.group(3)}"
    
    return None


def extract_summary(markdown: str, max_length: int = 300) -> str:
    """Extract a summary from markdown content."""
    if not markdown:
        return ""
    
    text = re.sub(r'^#+\s+.*$', '', markdown, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'[*_`]', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + '...'
    
    return text


def is_article_page(url: str, markdown: str) -> bool:
    """Determine if a page is likely an article vs. index/navigation page."""
    skip_patterns = [
        r'/tag/', r'/category/', r'/author/', r'/page/\d+',
        r'/archive', r'/search', r'/about', r'/contact',
        r'/privacy', r'/terms', r'/login', r'/register',
        r'/cart', r'/checkout', r'/account'
    ]
    
    url_lower = url.lower()
    for pattern in skip_patterns:
        if re.search(pattern, url_lower):
            return False
    
    if markdown:
        word_count = len(markdown.split())
        if word_count < 100:
            return False
    
    return True


async def crawl_source(
    url: str,
    pattern: Optional[str] = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_depth: int = DEFAULT_MAX_DEPTH
) -> dict:
    """Crawl a source URL and return discovered articles."""
    
    if not CRAWL4AI_AVAILABLE:
        return {
            "success": False,
            "articles": [],
            "pages_crawled": 0,
            "error": "Crawl4AI not installed"
        }
    
    start_time = time.time()
    articles = []
    pages_crawled = 0
    
    try:
        # Build filter chain
        filters = []
        if pattern and pattern.strip():
            # Auto-wrap each pattern with wildcards
            raw_patterns = [p.strip() for p in pattern.split(',')]
            patterns = [f"*{p}*" if not p.startswith('*') else p for p in raw_patterns]
            filters.append(URLPatternFilter(patterns=patterns))
        
        filter_chain = FilterChain(filters) if filters else None
        
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=False,
            filter_chain=filter_chain
        )
        
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        
        crawler_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl_strategy,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.4)
            ),
            excluded_tags=['nav', 'footer', 'aside', 'header'],
            remove_overlay_elements=True,
            process_iframes=False
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun(url, config=crawler_config)
            
            if not isinstance(results, list):
                results = [results]
            
            for result in results:
                pages_crawled += 1
                
                if not result.success:
                    continue
                
                markdown = ""
                if hasattr(result, 'markdown'):
                    if hasattr(result.markdown, 'fit_markdown'):
                        markdown = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
                    elif isinstance(result.markdown, str):
                        markdown = result.markdown
                
                if not is_article_page(result.url, markdown):
                    continue
                
                title = extract_title_from_markdown(markdown, result.url)
                article_date = extract_date_from_content(markdown, result.url)
                summary = extract_summary(markdown)
                
                articles.append({
                    "title": title,
                    "url": result.url,
                    "date": article_date,
                    "summary": summary,
                })
        
        duration = time.time() - start_time
        
        return {
            "success": True,
            "articles": articles,
            "pages_crawled": pages_crawled,
            "duration": duration,
            "error": None
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "articles": articles,
            "pages_crawled": pages_crawled,
            "duration": duration,
            "error": str(e)
        }

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

async def process_source(notion: Client, source: dict) -> dict:
    """Process a single source: crawl, dedupe, create entries."""
    
    logger.info(f"Processing: {source['name']} ({source['url']})")
    
    # Crawl the source
    crawl_result = await crawl_source(
        url=source["url"],
        pattern=source["crawl_pattern"],
        max_pages=source["max_pages"],
        max_depth=source["max_depth"]
    )
    
    if not crawl_result["success"]:
        logger.warning(f"  Crawl failed: {crawl_result['error']}")
        return {
            "source": source["name"],
            "success": False,
            "articles_found": 0,
            "articles_created": 0,
            "pages_crawled": crawl_result["pages_crawled"],
            "duration": crawl_result.get("duration", 0),
            "error": crawl_result["error"]
        }
    
    logger.info(f"  Crawled {crawl_result['pages_crawled']} pages, found {len(crawl_result['articles'])} potential articles")
    
    # Dedupe and create entries
    articles_created = 0
    topic = CATEGORY_TO_TOPIC.get(source["category"], "")
    
    for article in crawl_result["articles"]:
        # Check if URL already exists
        if url_exists_in_pipeline(notion, article["url"]):
            logger.debug(f"  Skipping (exists): {article['url']}")
            continue
        
        # Create new entry
        try:
            create_pipeline_entry(
                notion=notion,
                title=article["title"],
                url=article["url"],
                summary=article["summary"],
                topic=topic,
                source_name=source["name"]
            )
            articles_created += 1
            logger.info(f"  Created: {article['title'][:60]}...")
        except Exception as e:
            logger.error(f"  Failed to create entry for {article['url']}: {e}")
    
    logger.info(f"  Result: {articles_created} new articles created")
    
    return {
        "source": source["name"],
        "success": True,
        "articles_found": len(crawl_result["articles"]),
        "articles_created": articles_created,
        "pages_crawled": crawl_result["pages_crawled"],
        "duration": crawl_result.get("duration", 0),
        "error": None
    }


async def main(force_all: bool = False, dry_run: bool = False):
    """Main entry point."""
    
    logger.info("=" * 60)
    logger.info("Newsletter Crawler Starting")
    logger.info("=" * 60)
    
    # Determine which frequencies to process
    if force_all:
        frequencies = ["Daily", "Weekly", "Monthly", "Quarterly"]
        logger.info("Force mode: processing ALL frequencies")
    else:
        frequencies = get_frequencies_for_today()
        logger.info(f"Today's frequencies: {', '.join(frequencies)}")
    
    # Initialize Notion client
    notion = get_notion_client()
    
    # Query active sources
    sources = query_active_sources(notion, frequencies)
    logger.info(f"Found {len(sources)} active sources to process")
    
    if not sources:
        logger.info("No sources to process. Exiting.")
        return
    
    if dry_run:
        logger.info("\n--- DRY RUN MODE ---")
        for source in sources:
            logger.info(f"Would crawl: {source['name']}")
            logger.info(f"  URL: {source['url']}")
            logger.info(f"  Pattern: {source['crawl_pattern'] or '(none)'}")
            logger.info(f"  Max pages: {source['max_pages']}, Max depth: {source['max_depth']}")
            logger.info(f"  Frequency: {source['check_frequency']}")
        return
    
    # Process each source
    results = []
    for source in sources:
        result = await process_source(notion, source)
        results.append(result)
        
        # Update source status in Notion
        update_source_status(
            notion=notion,
            page_id=source["page_id"],
            success=result["success"],
            articles_found=result["articles_created"],  # Use created count (after dedup)
            pages_crawled=result["pages_crawled"],
            duration=result["duration"],
            error=result["error"],
            current_failures=source["consecutive_failures"]
        )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    total_created = sum(r["articles_created"] for r in results)
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    logger.info(f"Sources processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total new articles: {total_created}")
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        logger.info(f"  {status} {result['source']}: {result['articles_created']} new articles")
        if result["error"]:
            logger.info(f"      Error: {result['error'][:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Newsletter Pipeline Crawler")
    parser.add_argument("--force-all", action="store_true", 
                        help="Crawl all active sources regardless of frequency")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be crawled without doing it")
    
    args = parser.parse_args()
    
    asyncio.run(main(force_all=args.force_all, dry_run=args.dry_run))
