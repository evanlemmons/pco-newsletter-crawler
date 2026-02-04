#!/usr/bin/env python3
"""
monthly_summary.py - Generate monthly trend analysis from Newsletter Pipeline

This script:
1. Queries the past 30 days of articles from Newsletter Pipeline
2. Generates a strategic trend analysis using Claude Sonnet
3. Creates a Monthly Summary entry in Notion

Environment variables required:
- NOTION_API_KEY: Your Notion integration token
- ANTHROPIC_API_KEY: Your Anthropic API key

Usage:
    python monthly_summary.py              # Normal run
    python monthly_summary.py --dry-run    # Show what would be analyzed
    python monthly_summary.py --days 60    # Custom lookback period
"""

import argparse
import os
import sys
import logging
from datetime import datetime, date, timedelta
from calendar import monthrange
from collections import defaultdict

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
NEWSLETTER_PIPELINE_DB = "2efabbce69a280409309d052751eec14"

# Data Source IDs (for the new Notion API 2025-09-03)
NEWSLETTER_PIPELINE_DS = "2efabbce-69a2-8016-b362-000bfe5c9a11"

# ============================================================================
# NOTION CLIENT
# ============================================================================

try:
    from notion_client import Client
except ImportError:
    logger.error("notion-client not installed. Run: pip install notion-client")
    sys.exit(1)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.error("anthropic not installed. Run: pip install anthropic")
    sys.exit(1)


def get_anthropic_client():
    """Initialize Anthropic client with API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Cannot generate summary.")
        return None
    return anthropic.Anthropic(api_key=api_key)


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
# ARTICLE QUERIES
# ============================================================================

def query_recent_articles(notion: Client, start_date: date, end_date: date) -> list[dict]:
    """
    Query Newsletter Pipeline for Article entries in a date range.

    Args:
        start_date: Inclusive start date
        end_date: Exclusive end date (articles before this date)

    Returns list of dicts with: title, url, summary, topics, date_found, type
    """
    filter_query = {
        "and": [
            {"property": "Date Found", "date": {"on_or_after": start_date.isoformat()}},
            {"property": "Date Found", "date": {"before": end_date.isoformat()}},
            {"property": "Type", "select": {"equals": "Article"}}
        ]
    }

    articles = []
    has_more = True
    start_cursor = None

    while has_more:
        response = query_data_source(
            notion,
            NEWSLETTER_PIPELINE_DS,
            filter_query=filter_query,
            start_cursor=start_cursor
        )

        for page in response["results"]:
            props = page["properties"]

            # Extract title
            title = ""
            if props.get("Title", {}).get("title"):
                title = props["Title"]["title"][0]["plain_text"] if props["Title"]["title"] else ""

            # Extract URL
            url = props.get("URL", {}).get("url", "")

            # Extract summary
            summary = ""
            if props.get("Summary", {}).get("rich_text"):
                summary = props["Summary"]["rich_text"][0]["plain_text"] if props["Summary"]["rich_text"] else ""

            # Extract topics (multi-select)
            topics = []
            if props.get("Topic", {}).get("multi_select"):
                topics = [t["name"] for t in props["Topic"]["multi_select"]]

            # Extract date found
            date_found = ""
            if props.get("Date Found", {}).get("date"):
                date_found = props["Date Found"]["date"]["start"]

            # Extract type
            entry_type = ""
            if props.get("Type", {}).get("select"):
                entry_type = props["Type"]["select"]["name"]

            articles.append({
                "title": title,
                "url": url,
                "summary": summary,
                "topics": topics,
                "date_found": date_found,
                "type": entry_type,
            })

        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")

    return articles


# ============================================================================
# ARTICLE PROCESSING
# ============================================================================

def group_articles_by_topic(articles: list[dict]) -> dict[str, list]:
    """
    Group articles by their Topic multi-select.
    Articles with multiple topics appear in each group.
    Returns dict like: {"AI/ML": [...], "Church Tech": [...], "Uncategorized": [...]}
    """
    grouped = defaultdict(list)

    for article in articles:
        if article["topics"]:
            for topic in article["topics"]:
                grouped[topic].append(article)
        else:
            grouped["Uncategorized"].append(article)

    return dict(grouped)


def build_article_digest(articles: list[dict], max_chars_per_article: int = 300) -> str:
    """
    Build a condensed digest of all articles for Claude input.
    Includes URLs so Claude can cite sources in the output.

    Format:
    [AI/ML - 15 articles]
    - "Article Title" (URL) - Summary truncated...
    - "Another Article" (URL) - Summary...

    [Church Tech - 8 articles]
    ...
    """
    grouped = group_articles_by_topic(articles)

    lines = []

    # Sort topics by article count (most articles first)
    sorted_topics = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

    for topic, topic_articles in sorted_topics:
        lines.append(f"\n[{topic} - {len(topic_articles)} articles]")

        for article in topic_articles:
            title = article["title"][:100] if article["title"] else "(No title)"
            url = article["url"] or ""

            # Truncate summary but keep more detail
            summary = article["summary"] or "(No summary)"
            if len(summary) > max_chars_per_article:
                summary = summary[:max_chars_per_article].rsplit(' ', 1)[0] + "..."

            lines.append(f'- "{title}" ({url}) - {summary}')

    return "\n".join(lines)


# ============================================================================
# CLAUDE INTEGRATION
# ============================================================================

MONTHLY_SUMMARY_PROMPT = """You are a strategic analyst for Planning Center, a company that builds church management software used by thousands of churches. Your task is to write a monthly trend report based on articles from our industry monitoring. This should read like a polished internal newsletter for our product team.

CONTEXT:
- Planning Center builds tools for church operations: check-in, giving, groups, people management, services planning
- Our users are church administrators, pastors, and volunteers
- We monitor industry news, competitor activity, and user discussions to inform product decisions

ANALYSIS PERIOD: {period_description}
TOTAL ARTICLES ANALYZED: {article_count}

ARTICLE DIGEST (includes URLs for citation):
{article_digest}

CRITICAL: BE SELECTIVE, NOT COMPREHENSIVE

You are NOT expected to mention every article. Most articles in the digest are routine industry content that doesn't warrant inclusion. Your job is to identify and highlight ONLY the articles that:

1. Represent genuine strategic signals (not just marketing fluff or generic advice)
2. Indicate real industry shifts or competitive threats
3. Contain specific data, announcements, or insights that could inform product decisions
4. Show patterns when multiple sources discuss the same emerging issue

SKIP articles that are:
- Generic "how-to" content or best practices advice
- Marketing pieces without substantive news
- Repetitive coverage of the same topic (cite the best source, not all of them)
- Tangentially related but not actionable for Planning Center

If only 5-10 articles out of {article_count} are truly worth highlighting, that's fine. Quality over quantity.

Write a trend report with these exact section headers (including emojis):

## 📈 Emerging Trends

## 🏢 Major Industry Events

## 💬 User Sentiment & Pain Points

## 🎯 Strategic Implications

SECTION CONTENT REQUIREMENTS:

**📈 Emerging Trends**: Identify patterns where multiple sources discuss the same topic or theme. Only include if there's a genuine trend worth noting - not just because articles exist on a topic.

**🏢 Major Industry Events**: Cover significant events that would impact Planning Center - mentions of "Planning Center" specifically, acquisitions, mergers, partnerships, new product launches from competitors, new technologies being adopted by churches. If nothing major happened, say so briefly and move on.

**💬 User Sentiment & Pain Points**: What are church leaders and administrators talking about? Focus on pain points that Planning Center could address or should be aware of. Skip generic complaints.

**🎯 Strategic Implications**: Conclude with actionable insights for the product team. What should we pay attention to? What opportunities or threats are emerging? Be specific and tie back to the most important findings.

CRITICAL FORMATTING REQUIREMENT - TL;DR CALLOUTS:
Immediately after EACH section header (## emoji Title), include a TL;DR callout block using this exact format:

> **TL;DR:** [One short paragraph summarizing the 1-3 most important/impactful points from this section, including markdown links to the key articles]

Example:
## 📈 Emerging Trends

> **TL;DR:** AI adoption in churches has shifted from theoretical to practical, with [Barna research](url) showing 77% of pastors believe God can use AI. Meanwhile, the [Vanco-ACS merger](url) signals major consolidation in church software.

[Then continue with the full detailed section content...]

ADDITIONAL FORMATTING RULES:
- Use markdown formatting for readability (headers, bold, bullet points)
- Include URLs as markdown links: [Article Title](url)
- Write in a professional but accessible tone
- Be concise - a shorter, focused report is better than a long comprehensive one
- If a section has nothing noteworthy, include a brief "Nothing significant to report this month" and move on
- The TL;DR callouts are REQUIRED for each section

Begin your trend report:"""


def generate_monthly_summary(
    anthropic_client,
    article_digest: str,
    period_description: str,
    article_count: int
) -> str:
    """
    Generate strategic trend analysis using Claude Sonnet.

    Returns markdown-formatted analysis for Notion page body.
    """
    prompt = MONTHLY_SUMMARY_PROMPT.format(
        period_description=period_description,
        article_count=article_count,
        article_digest=article_digest
    )

    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        summary = message.content[0].text.strip()
        return summary

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise


# ============================================================================
# NOTION OUTPUT
# ============================================================================

def markdown_to_notion_blocks(markdown_text: str) -> list[dict]:
    """
    Convert markdown text to Notion blocks.
    Handles headers, paragraphs, bullet points, links, and callouts (blockquotes).
    """
    import re
    blocks = []
    lines = markdown_text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Blockquote / Callout (lines starting with >)
        if line.strip().startswith('>'):
            # Collect all consecutive blockquote lines
            callout_lines = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                # Remove the > prefix and any leading space
                content = lines[i].strip()[1:].strip()
                callout_lines.append(content)
                i += 1

            callout_text = ' '.join(callout_lines)

            blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": parse_inline_markdown(callout_text),
                    "color": "blue_background"
                }
            })
            continue

        # H1 header
        if line.startswith('# '):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": parse_inline_markdown(line[2:].strip())
                }
            })
        # H2 header
        elif line.startswith('## '):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": parse_inline_markdown(line[3:].strip())
                }
            })
        # H3 header
        elif line.startswith('### '):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": parse_inline_markdown(line[4:].strip())
                }
            })
        # Bullet point
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            text = line.strip()[2:]
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": parse_inline_markdown(text)
                }
            })
        # Regular paragraph
        else:
            # Collect consecutive non-header, non-bullet, non-blockquote lines into one paragraph
            paragraph_lines = [line]
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                if (next_line.strip() and
                    not next_line.startswith('#') and
                    not next_line.strip().startswith('- ') and
                    not next_line.strip().startswith('* ') and
                    not next_line.strip().startswith('>')):
                    paragraph_lines.append(next_line)
                    i += 1
                else:
                    break

            full_text = ' '.join(paragraph_lines)
            if full_text.strip():
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": parse_inline_markdown(full_text)
                    }
                })

        i += 1

    return blocks


def parse_inline_markdown(text: str) -> list[dict]:
    """
    Parse inline markdown (bold, links) into Notion rich_text array.
    """
    import re
    rich_text = []

    # Pattern to match markdown links [text](url) and bold **text**
    pattern = r'(\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))'
    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue

        # Bold text
        if part.startswith('**') and part.endswith('**'):
            rich_text.append({
                "type": "text",
                "text": {"content": part[2:-2]},
                "annotations": {"bold": True}
            })
        # Link
        elif part.startswith('[') and '](' in part:
            match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', part)
            if match:
                link_text, url = match.groups()
                rich_text.append({
                    "type": "text",
                    "text": {"content": link_text, "link": {"url": url}}
                })
        # Plain text
        else:
            rich_text.append({
                "type": "text",
                "text": {"content": part}
            })

    return rich_text if rich_text else [{"type": "text", "text": {"content": text}}]


def create_summary_entry(
    notion: Client,
    summary_text: str,
    month_name: str,
    article_count: int
) -> str:
    """
    Create Monthly Summary entry in Newsletter Pipeline.
    Puts the full content in the page body, not the Summary property.

    Returns page ID.
    """
    title = f"Monthly Summary - {month_name}"

    # Convert markdown to Notion blocks
    content_blocks = markdown_to_notion_blocks(summary_text)

    properties = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Type": {"select": {"name": "Monthly Summary"}},
        "Date Found": {"date": {"start": date.today().isoformat()}},
    }

    response = notion.pages.create(
        parent={"database_id": NEWSLETTER_PIPELINE_DB},
        icon={"type": "emoji", "emoji": "📣"},
        properties=properties,
        children=content_blocks
    )

    logger.info(f"Created Monthly Summary entry: {title}")
    return response["id"]


# ============================================================================
# MAIN
# ============================================================================

def get_previous_month_range() -> tuple[date, date, str]:
    """
    Calculate the date range for the previous month.
    Returns (start_date, end_date, month_name) where:
    - start_date: 1st of previous month
    - end_date: 1st of current month (exclusive end)
    - month_name: Name of the month being summarized (e.g., "January 2026")
    """
    today = date.today()
    # First day of current month
    first_of_current = today.replace(day=1)
    # First day of previous month
    if today.month == 1:
        first_of_previous = date(today.year - 1, 12, 1)
    else:
        first_of_previous = date(today.year, today.month - 1, 1)

    # Month name for the summary (the previous month)
    month_name = first_of_previous.strftime("%B %Y")

    return first_of_previous, first_of_current, month_name


def main(days_back: int = None, dry_run: bool = False):
    """Main entry point."""

    logger.info("=" * 60)
    logger.info("Monthly Summary Generator")
    logger.info("=" * 60)

    # Calculate period - always 1st of previous month to 1st of current month
    period_start, period_end, month_name = get_previous_month_range()
    period_description = f"{month_name} ({period_start.strftime('%B %d')} - {period_end.strftime('%B %d, %Y')})"

    # Allow override with --days for testing
    if days_back is not None:
        period_end = date.today()
        period_start = period_end - timedelta(days=days_back)
        period_description = f"{period_start.strftime('%B %d')} to {period_end.strftime('%B %d, %Y')} ({days_back} days)"
        month_name = period_end.strftime("%B %Y")

    logger.info(f"Analysis period: {period_description}")

    # Initialize clients
    notion = get_notion_client()
    anthropic_client = get_anthropic_client()

    if not anthropic_client:
        logger.error("Cannot proceed without Anthropic client")
        sys.exit(1)

    # Query recent articles
    logger.info("Querying Newsletter Pipeline for recent articles...")
    articles = query_recent_articles(notion, period_start, period_end)

    logger.info(f"Found {len(articles)} articles for {month_name}")

    if not articles:
        logger.info("No articles to analyze. Exiting.")
        return

    # Build article digest
    article_digest = build_article_digest(articles)

    # Show grouped counts
    grouped = group_articles_by_topic(articles)
    logger.info("Articles by topic:")
    for topic, topic_articles in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        logger.info(f"  {topic}: {len(topic_articles)}")

    if dry_run:
        logger.info("\n--- DRY RUN MODE ---")
        logger.info(f"Would analyze {len(articles)} articles")
        logger.info(f"Digest length: {len(article_digest)} characters")
        logger.info("\nSample of digest (first 2000 chars):")
        logger.info("-" * 40)
        print(article_digest[:2000])
        if len(article_digest) > 2000:
            logger.info(f"\n... and {len(article_digest) - 2000} more characters")
        return

    # Generate summary with Claude
    logger.info("Generating trend analysis with Claude Sonnet...")
    summary = generate_monthly_summary(
        anthropic_client,
        article_digest,
        period_description,
        len(articles)
    )

    logger.info("Summary generated successfully")
    logger.info("-" * 40)
    print(summary)
    logger.info("-" * 40)

    # Create Notion entry
    logger.info("Creating Monthly Summary entry in Notion...")
    page_id = create_summary_entry(
        notion,
        summary,
        month_name,
        len(articles)
    )

    logger.info(f"Done! Page ID: {page_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monthly trend analysis from Newsletter Pipeline")
    parser.add_argument("--days", type=int, default=None,
                        help="Override: look back N days instead of previous calendar month (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be analyzed without calling Claude or creating entries")

    args = parser.parse_args()

    main(days_back=args.days, dry_run=args.dry_run)
