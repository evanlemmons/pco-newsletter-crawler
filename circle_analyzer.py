#!/usr/bin/env python3
"""
circle_analyzer.py - Analyze Circle community discussions for themes and sentiment

This script:
1. Fetches posts and comments from all Circle community spaces
2. Aggregates content from the past 7 days (or since last crawl)
3. Uses Claude Sonnet to identify recurring themes, pain points, and sentiment
4. Creates thematic summary entries in Newsletter Pipeline
5. Updates Source Registry with analysis status

Environment variables required:
- NOTION_API_KEY: Your Notion integration token
- ANTHROPIC_API_KEY: Your Anthropic API key
- CIRCLE_API_KEY: Your Circle Admin API key

Usage:
    python circle_analyzer.py              # Normal run
    python circle_analyzer.py --dry-run    # Show what would be analyzed
    python circle_analyzer.py --days 30    # Custom lookback period
    python circle_analyzer.py --debug      # Verbose logging
"""

import argparse
import json
import os
import sys
import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional
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

# Circle API Configuration
CIRCLE_API_BASE = "https://app.circle.so/api/admin/v2"
REQUEST_DELAY = 0.5  # Seconds between API requests (rate limiting)

# Notion Database IDs
SOURCE_REGISTRY_DB = "43d593469bb8458c96ce927600514907"
NEWSLETTER_PIPELINE_DB = "2efabbce69a280409309d052751eec14"

# Data Source IDs (for the new Notion API 2025-09-03)
SOURCE_REGISTRY_DS = "902cd405-8998-4dc8-8810-ae8f54e3f61a"
NEWSLETTER_PIPELINE_DS = "2efabbce-69a2-8016-b362-000bfe5c9a11"

# Defaults
DEFAULT_DAYS_BACK = 7
MAX_THEMES_PER_RUN = 5

# ============================================================================
# DEPENDENCIES
# ============================================================================

try:
    import requests
except ImportError:
    logger.error("requests not installed. Run: pip install requests")
    sys.exit(1)

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

# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def get_circle_headers() -> dict:
    """Get Circle API headers with Bearer token."""
    api_key = os.environ.get("CIRCLE_API_KEY")
    if not api_key:
        logger.error("CIRCLE_API_KEY environment variable not set")
        sys.exit(1)

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


def get_anthropic_client():
    """Initialize Anthropic client with API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Cannot generate analysis.")
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
        kwargs["database_id"] = data_source_id
        del kwargs["data_source_id"]
        return notion.databases.query(**kwargs)

    raise RuntimeError("Notion SDK does not support data_sources.query or databases.query")

# ============================================================================
# CIRCLE API CLIENT
# ============================================================================

def make_circle_api_request(url: str, params: dict, headers: dict, max_retries: int = 3, debug: bool = False):
    """
    Make a Circle API request with retry logic and comprehensive error handling.

    Returns: Response data dict
    Raises: requests.exceptions.RequestException on failure after retries
    """
    for attempt in range(max_retries):
        try:
            time.sleep(REQUEST_DELAY)  # Rate limiting
            response = requests.get(url, headers=headers, params=params, timeout=30)

            # Handle rate limiting (429) with exponential backoff
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_after)
                continue

            # Handle server errors (5xx) with retry
            if response.status_code >= 500:
                wait_time = (2 ** attempt) * REQUEST_DELAY  # Exponential backoff
                logger.warning(f"Server error {response.status_code}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout. Retrying (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) * REQUEST_DELAY)
                continue
            raise

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1 and hasattr(e, 'response') and e.response is not None:
                status = e.response.status_code
                # Retry on specific error codes
                if status in [408, 429, 500, 502, 503, 504]:
                    wait_time = (2 ** attempt) * REQUEST_DELAY
                    logger.warning(f"API error {status}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

            # Log detailed error information
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API request failed with status {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise

    raise requests.exceptions.RequestException(f"Max retries ({max_retries}) exceeded")


def fetch_circle_spaces(debug: bool = False) -> list[dict]:
    """
    Fetch all community spaces from Circle API.

    Returns list of dicts with: id, name, slug, posts_count
    """
    headers = get_circle_headers()
    spaces = []
    page = 1

    try:
        while True:
            url = f"{CIRCLE_API_BASE}/spaces"
            params = {"page": page, "per_page": 100}

            if debug:
                logger.info(f"  [DEBUG] Fetching spaces page {page}: {url}")

            try:
                data = make_circle_api_request(url, params, headers, debug=debug)

                if debug:
                    logger.info(f"  [DEBUG] Response: {len(data.get('records', []))} spaces on this page")

                records = data.get("records", [])
                if not records:
                    break

                for space in records:
                    spaces.append({
                        "id": space.get("id"),
                        "name": space.get("name", ""),
                        "slug": space.get("slug", ""),
                        "posts_count": space.get("posts_count", 0),
                    })

                # Check pagination
                if not data.get("has_next_page", False):
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch spaces page {page}: {e}")
                raise

        logger.info(f"Successfully fetched {len(spaces)} spaces")
        return spaces

    except Exception as e:
        logger.error(f"Critical error fetching Circle spaces: {e}")
        logger.exception("Full traceback:")
        raise


def fetch_space_posts(space_id: int, since_date: datetime, debug: bool = False) -> list[dict]:
    """
    Fetch posts from a specific space since a given date.

    Returns list of dicts with: id, name, body, url, created_at, comments_count, likes_count
    Returns empty list on error (doesn't raise, to allow other spaces to continue)
    """
    headers = get_circle_headers()
    posts = []
    page = 1

    try:
        while True:
            url = f"{CIRCLE_API_BASE}/posts"
            params = {
                "space_id": space_id,
                "status": "published",
                "page": page,
                "per_page": 100
            }

            if debug:
                logger.info(f"  [DEBUG] Fetching posts for space {space_id}, page {page}")

            try:
                data = make_circle_api_request(url, params, headers, debug=debug)

                records = data.get("records", [])
                if not records:
                    break

                for post in records:
                    created_at_str = post.get("created_at", "")
                    try:
                        # Parse ISO timestamp
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

                        # Filter by date
                        if created_at < since_date:
                            # Posts are typically returned newest first, so we can stop
                            if debug:
                                logger.info(f"  [DEBUG] Reached posts older than {since_date}, stopping")
                            return posts

                        posts.append({
                            "id": post.get("id"),
                            "name": post.get("name", ""),
                            "body": post.get("body", ""),
                            "url": post.get("url", ""),
                            "created_at": created_at_str,
                            "comments_count": post.get("comments_count", 0),
                            "likes_count": post.get("likes_count", 0),
                        })

                    except (ValueError, TypeError) as e:
                        post_id = post.get('id', 'unknown')
                        logger.warning(f"Skipping post {post_id} in space {space_id}: Date parse error - {e}")
                        continue

                # Check pagination
                if not data.get("has_next_page", False):
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch posts page {page} for space {space_id}: {e}")
                # Return what we have so far instead of raising
                logger.warning(f"Returning {len(posts)} posts fetched before error")
                return posts

        return posts

    except Exception as e:
        logger.error(f"Unexpected error fetching posts for space {space_id}: {e}")
        logger.exception("Full traceback:")
        # Return empty list to allow other spaces to continue
        return []


def fetch_post_comments(post_id: int, debug: bool = False) -> list[dict]:
    """
    Fetch all comments for a specific post.

    Returns list of dicts with: id, body, created_at, user_name
    Returns empty list on error (comments are optional, shouldn't fail the whole process)
    """
    headers = get_circle_headers()
    comments = []
    page = 1

    try:
        while True:
            url = f"{CIRCLE_API_BASE}/comments"
            params = {
                "post_id": post_id,
                "page": page,
                "per_page": 100
            }

            if debug:
                logger.info(f"  [DEBUG] Fetching comments for post {post_id}, page {page}")

            try:
                data = make_circle_api_request(url, params, headers, max_retries=2, debug=debug)

                records = data.get("records", [])
                if not records:
                    break

                for comment in records:
                    try:
                        comments.append({
                            "id": comment.get("id"),
                            "body": comment.get("body", ""),
                            "created_at": comment.get("created_at", ""),
                            "user_name": comment.get("user", {}).get("name", "Unknown"),
                        })
                    except Exception as e:
                        logger.warning(f"Skipping malformed comment in post {post_id}: {e}")
                        continue

                # Check pagination
                if not data.get("has_next_page", False):
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch comments page {page} for post {post_id}: {e}")
                # Return what we have so far - comments are optional
                return comments

        return comments

    except Exception as e:
        logger.warning(f"Unexpected error fetching comments for post {post_id}: {e}")
        # Comments are optional - don't let this fail the analysis
        return []


# ============================================================================
# DATA AGGREGATION
# ============================================================================

def aggregate_space_content(space: dict, posts: list[dict], debug: bool = False) -> dict:
    """
    Aggregate posts and comments for a single space.

    Returns dict with: space_name, post_count, comment_count, content_digest
    """
    space_name = space["name"]

    if debug:
        logger.info(f"  [DEBUG] Aggregating {len(posts)} posts for space '{space_name}'")

    # Fetch comments for each post
    posts_with_comments = []
    total_comments = 0

    for post in posts:
        comments = fetch_post_comments(post["id"], debug=debug)
        posts_with_comments.append({
            "post": post,
            "comments": comments
        })
        total_comments += len(comments)

    # Build content digest for AI analysis
    digest_lines = []

    for item in posts_with_comments:
        post = item["post"]
        comments = item["comments"]

        # Post header
        digest_lines.append(f"\n## Post: {post['name']}")
        digest_lines.append(f"URL: {post['url']}")
        digest_lines.append(f"Created: {post['created_at']}")
        digest_lines.append(f"Likes: {post['likes_count']}, Comments: {post['comments_count']}")

        # Post body (truncate if very long)
        body = post["body"]
        if len(body) > 2000:
            body = body[:2000] + "... [truncated]"
        digest_lines.append(f"\nContent:\n{body}")

        # Comments
        if comments:
            digest_lines.append(f"\nComments ({len(comments)}):")
            for comment in comments:
                comment_body = comment["body"]
                if len(comment_body) > 500:
                    comment_body = comment_body[:500] + "..."
                digest_lines.append(f"- {comment['user_name']}: {comment_body}")

    content_digest = "\n".join(digest_lines)

    return {
        "space_name": space_name,
        "space_slug": space["slug"],
        "post_count": len(posts),
        "comment_count": total_comments,
        "content_digest": content_digest,
        "posts_with_comments": posts_with_comments
    }


# ============================================================================
# AI ANALYSIS
# ============================================================================

CIRCLE_ANALYSIS_PROMPT = """You are analyzing customer discussions from Planning Center's Circle community. Your goal is to identify significant themes, pain points, and sentiment patterns that would be valuable for the product team to understand.

CONTEXT:
- Planning Center builds church management software (check-in, giving, groups, people management, services planning)
- This is a private community where church admins, pastors, and staff discuss their experiences
- These discussions provide direct customer feedback, feature requests, and pain points

SPACE BEING ANALYZED: {space_name}
TIME PERIOD: Past {days_back} days
POSTS ANALYZED: {post_count}
COMMENTS ANALYZED: {comment_count}

CONTENT:
{content_digest}

YOUR TASK:
Identify recurring themes and patterns across these discussions. Focus ONLY on:

1. **Recurring themes**: Topics mentioned by multiple people or across multiple posts
2. **Pain points**: Specific frustrations, challenges, or problems customers are experiencing
3. **Feature requests**: Product capabilities customers are asking for
4. **Sentiment**: Overall tone and emotional patterns (enthusiasm, frustration, confusion, etc.)
5. **Customer quotes**: Extract powerful, specific quotes that really illustrate the theme

CRITICAL REQUIREMENTS FOR QUOTES:
- Extract ACTUAL quotes from the content (not paraphrased)
- Prioritize quotes that are specific, emotional, or particularly illustrative
- Include quotes that show real customer pain or enthusiasm
- Each quote should stand alone and be impactful

CRITICAL REQUIREMENTS FOR THREADS:
- Pay special attention to posts with many comments (5+ comments indicate high engagement)
- Prioritize threads where multiple people are discussing the same issue
- These high-engagement threads represent topics customers care deeply about

SKIP:
- One-off discussions that don't represent broader patterns
- Generic praise or complaints without specific details
- Admin/meta discussions about community management
- Only report themes that appear significant or actionable

If there are NO significant themes worth reporting in this space, respond with:
NO_SIGNIFICANT_THEMES

Otherwise, format your response as one or more theme blocks:

---THEME---
Title: [Concise theme name, e.g., "API Integration Challenges" or "Onboarding Confusion"]
Description: [2-3 sentences describing what people are discussing and why it matters]
Pain Points:
- [Specific challenge mentioned]
- [Another specific challenge]
Customer Quotes:
- "[Powerful direct quote from post or comment - be specific and emotional]" - [Author name if available]
- "[Another impactful quote showing the real customer experience]" - [Author name if available]
- "[Third quote if available - prioritize quotes that really illustrate the pain/need]" - [Author name if available]
Sentiment: [Positive/Neutral/Negative/Mixed + brief explanation]
Thread Engagement: [Note if any threads have many comments (5+) indicating high customer interest]
Recommended Actions: [1-2 specific suggestions for what Planning Center could do]
High-Impact Threads: [List post URLs for threads with highest engagement/most comments first]
Additional Thread Links: [Other related post URLs]
---END_THEME---

[Additional themes if there are more significant patterns...]

Remember: Quality over quantity. Extract quotes that would make a product manager sit up and take notice. Prioritize threads where customers are clearly passionate (lots of comments = people care about this topic)."""


def analyze_space_with_ai(
    anthropic_client,
    space_data: dict,
    days_back: int,
    debug: bool = False
) -> Optional[list[dict]]:
    """
    Analyze space content with Claude Sonnet to identify themes.

    Returns list of theme dicts, or None if no significant themes found.
    Each theme dict has: title, description, pain_points, quotes, sentiment, recommendations, source_urls
    """
    if not anthropic_client:
        logger.warning("No Anthropic client available, skipping AI analysis")
        return None

    space_name = space_data["space_name"]

    if debug:
        logger.info(f"  [DEBUG] Analyzing space '{space_name}' with Claude Sonnet")

    # Skip empty spaces
    if space_data["post_count"] == 0:
        if debug:
            logger.info(f"  [DEBUG] Skipping space '{space_name}' - no posts")
        return None

    prompt = CIRCLE_ANALYSIS_PROMPT.format(
        space_name=space_name,
        days_back=days_back,
        post_count=space_data["post_count"],
        comment_count=space_data["comment_count"],
        content_digest=space_data["content_digest"]
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

        response_text = message.content[0].text.strip()

        if debug:
            logger.info(f"  [DEBUG] AI response length: {len(response_text)} chars")

        # Check if no significant themes
        if "NO_SIGNIFICANT_THEMES" in response_text:
            logger.info(f"  No significant themes found in '{space_name}'")
            return None

        # Parse themes from response
        themes = parse_themes_from_response(response_text, space_name, debug=debug)

        if themes:
            logger.info(f"  Found {len(themes)} theme(s) in '{space_name}'")
        else:
            logger.info(f"  No parseable themes in '{space_name}' response")

        return themes

    except Exception as e:
        logger.error(f"Failed to analyze space '{space_name}': {e}")
        if debug:
            logger.exception("Full traceback:")
        return None


def parse_themes_from_response(response_text: str, space_name: str, debug: bool = False) -> list[dict]:
    """
    Parse theme blocks from AI response.

    Expected format:
    ---THEME---
    Title: ...
    Description: ...
    Pain Points:
    - ...
    Customer Quotes:
    - ...
    Sentiment: ...
    Thread Engagement: ...
    Recommended Actions: ...
    High-Impact Threads: ...
    Additional Thread Links: ...
    ---END_THEME---
    """
    themes = []

    # Split by theme markers
    theme_blocks = response_text.split("---THEME---")

    for block in theme_blocks[1:]:  # Skip first empty block
        if "---END_THEME---" not in block:
            continue

        # Extract content between markers
        theme_content = block.split("---END_THEME---")[0].strip()

        # Parse fields
        theme = {
            "space_name": space_name,
            "title": "",
            "description": "",
            "pain_points": [],
            "quotes": [],
            "sentiment": "",
            "thread_engagement": "",
            "recommendations": "",
            "high_impact_urls": [],
            "additional_urls": []
        }

        current_field = None
        lines = theme_content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for field headers
            if line.startswith("Title:"):
                theme["title"] = line[6:].strip()
                current_field = None
            elif line.startswith("Description:"):
                theme["description"] = line[12:].strip()
                current_field = "description"
            elif line.startswith("Pain Points:"):
                current_field = "pain_points"
            elif line.startswith("Customer Quotes:") or line.startswith("Quotes:"):
                current_field = "quotes"
            elif line.startswith("Sentiment:"):
                theme["sentiment"] = line[10:].strip()
                current_field = "sentiment"
            elif line.startswith("Thread Engagement:"):
                theme["thread_engagement"] = line[18:].strip()
                current_field = "thread_engagement"
            elif line.startswith("Recommended Actions:"):
                theme["recommendations"] = line[20:].strip()
                current_field = "recommendations"
            elif line.startswith("High-Impact Threads:"):
                current_field = "high_impact_urls"
            elif line.startswith("Additional Thread Links:") or line.startswith("Source Posts:"):
                current_field = "additional_urls"
            # Handle list items
            elif line.startswith("- "):
                content = line[2:].strip()
                if current_field == "pain_points":
                    theme["pain_points"].append(content)
                elif current_field == "quotes":
                    # Remove quote marks if present
                    content = content.strip('"').strip("'")
                    theme["quotes"].append(content)
                elif current_field == "high_impact_urls":
                    # Extract URLs
                    if content.startswith("http"):
                        theme["high_impact_urls"].append(content)
                elif current_field == "additional_urls":
                    # Extract URLs
                    if content.startswith("http"):
                        theme["additional_urls"].append(content)
            # Handle continuation lines
            elif current_field in ["description", "sentiment", "thread_engagement", "recommendations"]:
                theme[current_field] += " " + line

        # Combine high-impact and additional URLs into source_urls for backward compatibility
        theme["source_urls"] = theme["high_impact_urls"] + theme["additional_urls"]

        # Only add theme if it has a title and description
        if theme["title"] and theme["description"]:
            themes.append(theme)
            if debug:
                logger.info(f"  [DEBUG] Parsed theme: {theme['title']}")
                logger.info(f"  [DEBUG]   Quotes: {len(theme['quotes'])}, High-impact threads: {len(theme['high_impact_urls'])}")
        else:
            if debug:
                logger.info(f"  [DEBUG] Skipping incomplete theme block")

    return themes


# ============================================================================
# NOTION INTEGRATION
# ============================================================================

def query_circle_source(notion: Client) -> Optional[dict]:
    """
    Query Source Registry for Circle Community entry.
    Returns dict with: page_id, name, last_crawl_date, or None if not found.
    """
    # Query for "Circle Community" by name
    filter_query = {
        "and": [
            {"property": "Name", "title": {"contains": "Circle"}},
            {"property": "Status", "select": {"equals": "Active"}}
        ]
    }

    response = query_data_source(
        notion,
        SOURCE_REGISTRY_DS,
        filter_query=filter_query,
        page_size=1
    )

    if not response["results"]:
        logger.warning("Circle Community source not found in Source Registry")
        return None

    page = response["results"][0]
    props = page["properties"]

    # Extract last success date if available
    last_crawl_date = None
    if props.get("Last Success", {}).get("date"):
        last_crawl_str = props["Last Success"]["date"]["start"]
        try:
            last_crawl_date = datetime.fromisoformat(last_crawl_str).date()
        except (ValueError, TypeError):
            pass

    name = ""
    if props.get("Name", {}).get("title"):
        name = props["Name"]["title"][0]["plain_text"] if props["Name"]["title"] else ""

    return {
        "page_id": page["id"],
        "name": name,
        "last_crawl_date": last_crawl_date
    }


def create_theme_entry(
    notion: Client,
    theme: dict,
    source_page_id: str,
    debug: bool = False
) -> str:
    """
    Create a Newsletter Pipeline entry for a Circle theme.

    Returns page ID.
    """
    # Build title
    title = f"Circle: {theme['title']}"
    if len(title) > 100:
        title = title[:97] + "..."

    # Build rich summary with emphasis on customer quotes and threads
    summary_parts = [
        f"Space: {theme['space_name']}",
        f"\n\n{theme['description']}",
    ]

    # Add thread engagement if available
    if theme.get("thread_engagement"):
        summary_parts.append(f"\n\n🔥 Thread Activity: {theme['thread_engagement']}")

    # Add pain points
    if theme["pain_points"]:
        summary_parts.append("\n\n💔 Pain Points:")
        for point in theme["pain_points"][:3]:  # Limit to 3
            summary_parts.append(f"\n• {point}")

    # Add customer quotes (multiple if available)
    if theme["quotes"]:
        summary_parts.append("\n\n💬 Customer Voices:")
        for i, quote in enumerate(theme["quotes"][:3]):  # Limit to 3 quotes
            # Truncate very long quotes
            if len(quote) > 200:
                quote = quote[:197] + "..."
            summary_parts.append(f"\n\"{quote}\"")
            if i < len(theme["quotes"]) - 1:  # Add separator if not last quote
                summary_parts.append("\n")

    # Add sentiment
    summary_parts.append(f"\n\n😊 Sentiment: {theme['sentiment']}")

    # Add high-impact thread links
    if theme.get("high_impact_urls"):
        summary_parts.append("\n\n🔗 High-Engagement Threads:")
        for url in theme["high_impact_urls"][:2]:  # Limit to 2
            summary_parts.append(f"\n• {url}")

    # Add recommendations
    if theme["recommendations"]:
        summary_parts.append(f"\n\n💡 Recommended Actions:\n{theme['recommendations']}")

    summary = "".join(summary_parts)

    # Truncate if too long (prioritize keeping quotes and links)
    if len(summary) > 1900:
        # Try to keep at least description, quotes, and first link
        summary = summary[:1900] + "..."

    # Determine topic based on content
    topic = "Church Tech"  # Default
    description_lower = theme["description"].lower()
    if any(word in description_lower for word in ["product", "feature", "roadmap", "planning"]):
        topic = "Product Management"

    properties = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Type": {"select": {"name": "Article"}},  # Using Article since no Circle-specific type
        "Topic": {"multi_select": [{"name": topic}]},
        "Date Found": {"date": {"start": date.today().isoformat()}},
        "Summary": {"rich_text": [{"text": {"content": summary}}]},
        "Source Page": {"relation": [{"id": source_page_id}]},
        "Status": {"select": {"name": "Unreviewed"}},
    }

    # Use first high-impact URL if available, otherwise fall back to any source URL
    if theme.get("high_impact_urls"):
        properties["URL"] = {"url": theme["high_impact_urls"][0]}
    elif theme["source_urls"]:
        properties["URL"] = {"url": theme["source_urls"][0]}

    response = notion.pages.create(
        parent={"database_id": NEWSLETTER_PIPELINE_DB},
        icon={"type": "emoji", "emoji": "💬"},
        properties=properties
    )

    if debug:
        logger.info(f"  [DEBUG] Created entry: {title}")

    return response["id"]


def update_source_status(
    notion: Client,
    page_id: str,
    success: bool,
    themes_created: int,
    spaces_analyzed: int,
    posts_analyzed: int,
    duration: float,
    error: Optional[str]
) -> None:
    """Update Circle source entry in Source Registry with analysis results."""

    today_iso = date.today().isoformat()

    properties = {
        "Last Reviewed": {"date": {"start": today_iso}},
    }

    if success:
        properties["Last Success"] = {"date": {"start": today_iso}}
        properties["Consecutive Failures"] = {"number": 0}
        properties["Crawl Notes"] = {
            "rich_text": [{"text": {"content":
                f"Analyzed {spaces_analyzed} spaces, {posts_analyzed} posts. Found {themes_created} significant themes. Duration: {duration:.1f}s"
            }}]
        }
    else:
        error_msg = error or "Unknown error"
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."

        properties["Crawl Notes"] = {
            "rich_text": [{"text": {"content": f"Analysis failed: {error_msg}"}}]
        }

    notion.pages.update(page_id=page_id, properties=properties)


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main(days_back: int = DEFAULT_DAYS_BACK, dry_run: bool = False, debug: bool = False):
    """Main entry point."""

    logger.info("=" * 60)
    logger.info("Circle Community Analyzer")
    logger.info("=" * 60)

    if debug:
        logger.info("DEBUG MODE ENABLED")

    start_time = time.time()

    # Calculate since_date
    since_date = datetime.now() - timedelta(days=days_back)
    logger.info(f"Analyzing content from past {days_back} days (since {since_date.strftime('%Y-%m-%d')})")

    # Initialize clients
    notion = get_notion_client()
    anthropic_client = get_anthropic_client()

    if not anthropic_client:
        logger.error("Cannot proceed without Anthropic client")
        sys.exit(1)

    # Query Circle source from Notion
    logger.info("Querying Source Registry for Circle Community...")
    circle_source = query_circle_source(notion)

    if not circle_source:
        logger.error("Circle Community not found in Source Registry. Please add it first.")
        logger.error("See plan for instructions on creating the Source Registry entry.")
        sys.exit(1)

    logger.info(f"Found Circle source: {circle_source['name']}")

    # Fetch spaces
    logger.info("Fetching Circle community spaces...")
    try:
        spaces = fetch_circle_spaces(debug=debug)
        logger.info(f"Found {len(spaces)} spaces")
    except Exception as e:
        logger.error(f"Failed to fetch spaces: {e}")
        update_source_status(
            notion,
            circle_source["page_id"],
            success=False,
            themes_created=0,
            spaces_analyzed=0,
            posts_analyzed=0,
            duration=time.time() - start_time,
            error=str(e)
        )
        sys.exit(1)

    if dry_run:
        logger.info("\n--- DRY RUN MODE ---")
        for space in spaces:
            logger.info(f"Would analyze space: {space['name']} ({space['posts_count']} total posts)")
        return

    # Analyze each space
    all_themes = []
    total_posts = 0
    spaces_with_errors = []
    spaces_processed = 0

    for i, space in enumerate(spaces, 1):
        logger.info(f"\n[{i}/{len(spaces)}] Processing space: {space['name']}")

        try:
            # Fetch posts
            logger.info(f"  Fetching posts from past {days_back} days...")
            posts = fetch_space_posts(space["id"], since_date, debug=debug)
            logger.info(f"  Found {len(posts)} recent posts")

            if not posts:
                logger.info(f"  Skipping space (no recent posts)")
                continue

            total_posts += len(posts)
            spaces_processed += 1

            # Aggregate content
            logger.info(f"  Aggregating posts and comments...")
            try:
                space_data = aggregate_space_content(space, posts, debug=debug)
                logger.info(f"  Total: {space_data['post_count']} posts, {space_data['comment_count']} comments")
            except Exception as e:
                logger.error(f"  Failed to aggregate content: {e}")
                spaces_with_errors.append((space['name'], f"Aggregation error: {str(e)[:100]}"))
                continue

            # AI analysis
            logger.info(f"  Analyzing with Claude Sonnet...")
            try:
                themes = analyze_space_with_ai(
                    anthropic_client,
                    space_data,
                    days_back,
                    debug=debug
                )

                if themes:
                    all_themes.extend(themes)
                    for theme in themes:
                        logger.info(f"    ✓ Theme: {theme['title']}")
            except Exception as e:
                logger.error(f"  Failed to analyze with AI: {e}")
                spaces_with_errors.append((space['name'], f"AI analysis error: {str(e)[:100]}"))
                if debug:
                    logger.exception("Full traceback:")
                continue

        except Exception as e:
            logger.error(f"  Failed to process space '{space['name']}': {e}")
            spaces_with_errors.append((space['name'], str(e)[:100]))
            if debug:
                logger.exception("Full traceback:")
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Spaces found: {len(spaces)}")
    logger.info(f"Spaces successfully processed: {spaces_processed}")
    logger.info(f"Spaces with errors: {len(spaces_with_errors)}")
    logger.info(f"Posts analyzed: {total_posts}")
    logger.info(f"Themes identified: {len(all_themes)}")

    if spaces_with_errors:
        logger.warning("\nSpaces that encountered errors:")
        for space_name, error in spaces_with_errors:
            logger.warning(f"  • {space_name}: {error}")

    if not all_themes:
        logger.info("No significant themes found. No entries will be created.")
        update_source_status(
            notion,
            circle_source["page_id"],
            success=True,
            themes_created=0,
            spaces_analyzed=len(spaces),
            posts_analyzed=total_posts,
            duration=time.time() - start_time,
            error=None
        )
        return

    # Limit themes if too many
    if len(all_themes) > MAX_THEMES_PER_RUN:
        logger.info(f"Limiting to top {MAX_THEMES_PER_RUN} themes")
        all_themes = all_themes[:MAX_THEMES_PER_RUN]

    # Create Notion entries
    logger.info(f"\nCreating {len(all_themes)} Newsletter Pipeline entries...")
    themes_created = 0

    for theme in all_themes:
        try:
            create_theme_entry(
                notion,
                theme,
                circle_source["page_id"],
                debug=debug
            )
            themes_created += 1
            logger.info(f"  ✓ Created: {theme['title']}")
        except Exception as e:
            logger.error(f"  ✗ Failed to create entry for '{theme['title']}': {e}")
            if debug:
                logger.exception("Full traceback:")

    # Update source status
    update_source_status(
        notion,
        circle_source["page_id"],
        success=True,
        themes_created=themes_created,
        spaces_analyzed=len(spaces),
        posts_analyzed=total_posts,
        duration=time.time() - start_time,
        error=None
    )

    logger.info(f"\n✓ Done! Created {themes_created} theme entries in Newsletter Pipeline")
    logger.info(f"Total duration: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circle Community Analyzer")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK,
                        help=f"Look back N days (default: {DEFAULT_DAYS_BACK})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be analyzed without calling AI or creating entries")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")

    args = parser.parse_args()

    main(days_back=args.days, dry_run=args.dry_run, debug=args.debug)
