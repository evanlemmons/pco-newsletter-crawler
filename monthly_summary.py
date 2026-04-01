#!/usr/bin/env python3
"""
monthly_summary.py - Generate monthly trend analysis from Newsletter Pipeline

This script:
1. Queries the past 30 days of articles from Newsletter Pipeline
2. Generates a strategic trend analysis using Claude Sonnet
3. Creates a Monthly Summary entry in Notion
4. Posts notification to Slack with headline and link

Environment variables required:
- NOTION_API_KEY: Your Notion integration token
- ANTHROPIC_API_KEY: Your Anthropic API key
- SLACK_WEBHOOK_URL: Slack incoming webhook for #product-department (optional)

Usage:
    python monthly_summary.py              # Normal run
    python monthly_summary.py --dry-run    # Show what would be analyzed
    python monthly_summary.py --days 60    # Custom lookback period
"""

import argparse
import json
import os
import sys
import logging
import urllib.request
import urllib.error
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
    Query Newsletter Pipeline for Article and Community Discussion entries in a date range.

    Args:
        start_date: Inclusive start date
        end_date: Exclusive end date (articles before this date)

    Returns list of dicts with: title, url, summary, topics, date_found, type
    """
    filter_query = {
        "and": [
            {"property": "Date Found", "date": {"on_or_after": start_date.isoformat()}},
            {"property": "Date Found", "date": {"before": end_date.isoformat()}},
            {
                "or": [
                    {"property": "Type", "select": {"equals": "Article"}},
                    {"property": "Type", "select": {"equals": "Community Discussion"}}
                ]
            }
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

            # Extract summary (may be stored across multiple rich_text chunks)
            summary = ""
            if props.get("Summary", {}).get("rich_text"):
                summary = "".join(
                    chunk.get("plain_text", chunk.get("text", {}).get("content", ""))
                    for chunk in props["Summary"]["rich_text"]
                )

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


def build_article_digest(articles: list[dict]) -> str:
    """
    Build a digest of all articles for Claude input.
    Includes URLs so Claude can cite sources in the output.

    Separates Community Discussion entries from regular Articles.

    Format:
    [ARTICLES BY TOPIC]
    [AI/ML - 15 articles]
    - "Article Title" (URL) - Summary...

    [COMMUNITY DISCUSSIONS]
    - "Discussion Title" (URL) - Summary...
    """
    # Separate articles by type
    regular_articles = [a for a in articles if a["type"] == "Article"]
    community_discussions = [a for a in articles if a["type"] == "Community Discussion"]

    lines = []

    # Process regular articles grouped by topic
    if regular_articles:
        lines.append("\n[ARTICLES BY TOPIC]")
        grouped = group_articles_by_topic(regular_articles)
        sorted_topics = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

        for topic, topic_articles in sorted_topics:
            lines.append(f"\n[{topic} - {len(topic_articles)} articles]")

            for article in topic_articles:
                title = article["title"][:100] if article["title"] else "(No title)"
                url = article["url"] or ""
                summary = article["summary"] or "(No summary)"
                lines.append(f'- "{title}" ({url}) - {summary}')

    # Process community discussions separately
    if community_discussions:
        lines.append(f"\n\n[COMMUNITY DISCUSSIONS - {len(community_discussions)} entries]")
        for discussion in community_discussions:
            title = discussion["title"][:100] if discussion["title"] else "(No title)"
            url = discussion["url"] or ""
            summary = discussion["summary"] or "(No summary)"
            lines.append(f'- "{title}" ({url}) - {summary}')

    return "\n".join(lines)


# ============================================================================
# CLAUDE INTEGRATION
# ============================================================================

MONTHLY_SUMMARY_PROMPT = """You are a strategic analyst for Planning Center, a company that builds church management software used by thousands of churches. Your task is to write a monthly trend report based on articles from our industry monitoring. This should read like a polished internal newsletter for our product team.

FIRST, OUTPUT A CLICKBAIT HEADLINE:
Before the main report, output a single catchy, clickbait-style, semi-humorous one-liner that captures the month's most interesting theme or finding. This will be used as a teaser summary.

Format it exactly like this on its own line:
HEADLINE: [Your snappy one-liner here]

Examples of good headlines:
- HEADLINE: AI is coming for your church bulletin, and pastors are surprisingly okay with it
- HEADLINE: Three ChMS vendors walk into a merger, only two walk out
- HEADLINE: Turns out churches DO want online giving—who knew?
- HEADLINE: The great volunteer shortage of 2026 is real, and it's spectacular

The headline should be:
- One sentence, under 100 characters ideally
- Slightly irreverent but still professional
- Reference a specific finding from this month's articles
- Make someone want to read more

After the headline, continue with the full report below.

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

Write a trend report with these section headers (including emojis):

## 📈 Emerging Trends

## 🏢 Major Industry Events

## 💬 User Sentiment & Pain Points

## 💭 Community Discussions
[ONLY include this section if there are Community Discussion entries in the digest AND they contain substantial insights worth highlighting. If community discussions are trivial or redundant with other sections, skip this section entirely.]

## 🎯 Strategic Implications

SECTION CONTENT REQUIREMENTS:

**📈 Emerging Trends**: Identify patterns where multiple sources discuss the same topic or theme. Only include if there's a genuine trend worth noting - not just because articles exist on a topic.

**🏢 Major Industry Events**: Cover significant events that would impact Planning Center - mentions of "Planning Center" specifically, acquisitions, mergers, partnerships, new product launches from competitors, new technologies being adopted by churches. If nothing major happened, say so briefly and move on.

**💬 User Sentiment & Pain Points**: What are church leaders and administrators talking about? Focus on pain points that Planning Center could address or should be aware of. Skip generic complaints.

**💭 Community Discussions**: [CONDITIONAL SECTION] If the digest includes Community Discussion entries with substantial insights, create this section to highlight key themes, pain points, or feature requests from the Planning Center community. This section should focus on direct user feedback and discussions that differ from external industry content. Do NOT duplicate information that's already covered in other sections. If community discussions don't add unique value beyond what's in other sections, omit this section entirely.

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


SECTION_SYSTEM_CONTEXT = """You are a strategic analyst for Planning Center, a company that builds church management software used by thousands of churches. You are writing one section of a monthly internal trend newsletter for the product team.

CONTEXT:
- Planning Center builds tools for church operations: check-in, giving, groups, people management, services planning
- Our users are church administrators, pastors, and volunteers
- We monitor industry news, competitor activity, and user discussions to inform product decisions

ANALYSIS PERIOD: {period_description}
TOTAL ARTICLES IN DIGEST: {article_count}

ARTICLE DIGEST:
{article_digest}
"""

SECTION_PROMPTS = {
    "trends": """Write ONLY the ## 📈 Emerging Trends section of the newsletter.

Identify patterns where multiple sources discuss the same topic. Only include genuine trends — not just because articles exist on a topic. Skip generic content.

Start with this exact format:
## 📈 Emerging Trends

> **TL;DR:** [2 sentences max — the 1-3 most important trend signals, including markdown links]

Then write the detailed section using ### subheadings for each major trend. Under each ### subheading, lead with 1-2 sentences of prose context, then use - bullet points only if you have 3 or more parallel items to list. Aim for a natural mix — not all bullets, not all prose.

Style rules:
- Target 350-450 words for the full section
- Use markdown links: [Title](url)
- Bold ONLY statistics and key data points (e.g. **77%**, **$2.4M**) — not headers, not link text
- Do NOT output --- horizontal rules
- Do not use # or ## headers""",

    "events": """Write ONLY the ## 🏢 Major Industry Events section of the newsletter.

Cover significant events impacting Planning Center: direct mentions of "Planning Center", competitor acquisitions/mergers/launches, new technologies being adopted by churches. If nothing major happened, say so in one sentence and stop.

Start with this exact format:
## 🏢 Major Industry Events

> **TL;DR:** [2 sentences max — the 1-2 most important events, including markdown links]

Then write the detailed section using ### subheadings for each event or theme. Under each ### subheading, lead with prose sentences — use - bullet points only if you have 3 or more parallel sub-points to list.

Style rules:
- Target 300-400 words for the full section
- Use markdown links: [Title](url)
- Bold ONLY statistics and key data points — not headers, not link text
- Do NOT output --- horizontal rules
- If truly nothing noteworthy happened, write only: "Nothing significant to report this month."
- Do not use # or ## headers""",

    "sentiment": """Write ONLY the ## 💬 User Sentiment & Pain Points section of the newsletter.

Focus on what church leaders and administrators are talking about — pain points Planning Center could address or should be aware of. Skip generic complaints. Only highlight if there's a genuine signal worth acting on.

Start with this exact format:
## 💬 User Sentiment & Pain Points

> **TL;DR:** [2 sentences max — the 1-2 most actionable pain points, including markdown links]

Then write the detailed section using ### subheadings for each pain point or theme. Under each ### subheading, write 2-3 sentences of prose explaining the issue and why it matters — avoid converting everything into bullets. Use - bullet points only if listing 3+ concrete examples or symptoms.

Style rules:
- Target 300-400 words for the full section
- Use markdown links: [Title](url)
- Bold ONLY statistics and key data points — not headers, not link text
- Do NOT output --- horizontal rules
- Do not use # or ## headers""",

    "community": """Write ONLY the ## 💭 Community Discussions section of the newsletter.

You are receiving Community Discussion entries from our Circle community. Each entry's summary starts with 🔗 Conversation Links — use those URLs as inline markdown citations when referencing specific themes: [brief label](url).

Focus on: recurring themes, pain points, feature requests from actual Planning Center users. This should feel different from external industry content — it's direct user voice.

If the entries don't contain substantial unique insights, output exactly: SKIP_SECTION

Otherwise start with this exact format:
## 💭 Community Discussions

> **TL;DR:** [2 sentences max — the 1-2 most important community signals, including markdown links to specific conversations]

Write a brief intro paragraph (2-3 sentences) framing the overall community mood this month.

Then use ### subheadings for each major theme, prefixed with a priority emoji:
- 🔴 for high-priority pain points (multiple users, impacts core workflows)
- 🟡 for medium-priority issues (notable but less urgent)
- 🟢 for opportunities and positive signals

Under each ### subheading, write 2-3 sentences of prose — cite conversation links inline. Use - bullet points only for listing 3+ concrete workarounds or examples.

Style rules:
- Target 350-450 words for the full section
- Bold ONLY statistics and key data points — not headers, not link text
- Do NOT output --- horizontal rules
- Do not use # or ## headers""",

    "implications": """Write ONLY the ## 🎯 Strategic Implications section of the newsletter.

You are given the other sections of this month's newsletter as context. Synthesize them into actionable insights for the product team. Be specific — tie back to named findings from the other sections.

{previous_sections}

Start with this exact format:
## 🎯 Strategic Implications

> **TL;DR:** [2 sentences max — the 2-3 most important strategic takeaways, including markdown links]

Write 3-5 - bullet points. Each bullet should name the signal and state the implication for Planning Center in 2-3 sentences. Do not use ### subheadings.

Style rules:
- Target 200-300 words total
- Bold ONLY statistics and key data points — not headers, not link text
- Do NOT output --- horizontal rules
- Do not use # or ## headers""",

    "headline": """Based on this month's newsletter sections below, write a single clickbait headline.

{all_sections}

Output ONLY this line (nothing else):
HEADLINE: [Your snappy one-liner here]

Requirements:
- One sentence, under 100 characters
- Slightly irreverent but professional
- References a specific finding from this month's content
- Makes someone want to read more

Examples:
- HEADLINE: AI is coming for your church bulletin, and pastors are surprisingly okay with it
- HEADLINE: Three ChMS vendors walk into a merger, only two walk out"""
}


def generate_section(
    section_key: str,
    digest: str,
    period_description: str,
    article_count: int,
    model: str = "claude-opus-4-6",
    previous_sections: str = "",
    all_sections: str = "",
) -> str:
    """Generate a single newsletter section using a focused prompt."""
    import anthropic
    client = anthropic.Anthropic()

    system_context = SECTION_SYSTEM_CONTEXT.format(
        period_description=period_description,
        article_count=article_count,
        article_digest=digest,
    )

    section_prompt = SECTION_PROMPTS[section_key]
    if section_key == "implications":
        section_prompt = section_prompt.format(previous_sections=previous_sections)
    elif section_key == "headline":
        section_prompt = section_prompt.format(all_sections=all_sections)
        # Headline doesn't need the article digest in system context
        system_context = ""

    full_prompt = f"{system_context}\n\n{section_prompt}" if system_context else section_prompt

    message = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": full_prompt}],
    )

    return message.content[0].text.strip()


def generate_monthly_summary(
    anthropic_client,
    article_digest: str,
    period_description: str,
    article_count: int
) -> tuple[str, str]:
    """
    Generate strategic trend analysis using per-section Claude Opus calls.

    Returns tuple of (headline, body) where:
    - headline: Clickbait-style one-liner for the Summary property
    - body: Full markdown-formatted analysis for Notion page body
    """
    try:
        # Separate community discussions from regular articles
        articles = []  # article_digest is pre-built; routing is handled in generate_section
        community_discussions = []

        shared_kwargs = dict(
            period_description=period_description,
            article_count=article_count,
            model="claude-opus-4-6",
        )

        print("Generating section: Emerging Trends...")
        trends = generate_section("trends", article_digest, **shared_kwargs)

        print("Generating section: Major Industry Events...")
        events = generate_section("events", article_digest, **shared_kwargs)

        print("Generating section: User Sentiment & Pain Points...")
        sentiment = generate_section("sentiment", article_digest, **shared_kwargs)

        # Community section uses only the community portion of the digest.
        # build_article_digest already groups community discussions under
        # [COMMUNITY DISCUSSIONS] — pass the full digest and let the prompt
        # focus the model on that block.
        print("Generating section: Community Discussions...")
        community_raw = generate_section("community", article_digest, **shared_kwargs)
        community = "" if community_raw.strip() == "SKIP_SECTION" else community_raw

        # Strategic Implications sees all prior sections
        previous_sections_text = "\n\n---\n\n".join(filter(None, [trends, events, sentiment, community]))
        print("Generating section: Strategic Implications...")
        implications = generate_section(
            "implications",
            article_digest,
            previous_sections=previous_sections_text,
            **shared_kwargs,
        )

        # Assemble body
        sections = [trends, events, sentiment]
        if community:
            sections.append(community)
        sections.append(implications)
        all_sections_text = "\n\n---\n\n".join(sections)

        # Generate headline
        print("Generating headline...")
        headline_raw = generate_section(
            "headline",
            digest="",
            period_description=period_description,
            article_count=article_count,
            model="claude-opus-4-6",
            all_sections=all_sections_text,
        )
        headline = headline_raw.replace("HEADLINE:", "").strip()
        if not headline:
            logger.warning("No HEADLINE found in response, using default")
            headline = f"What happened in {period_description}"
        else:
            logger.info(f"Extracted headline: {headline}")

        body = all_sections_text
        return headline, body

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

        # Horizontal rule → Notion divider
        if line.strip() == '---':
            blocks.append({"object": "block", "type": "divider", "divider": {}})
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

    # Pattern to match: bold links **[text](url)**, plain links [text](url), bold **text**
    # Bold links must come first — otherwise the bold pattern swallows the link syntax
    pattern = r'(\*\*\[[^\]]+\]\([^)]+\)\*\*|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))'
    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue

        # Bold link: **[text](url)**
        if part.startswith('**[') and part.endswith(')**'):
            inner = part[2:-2]  # strip surrounding **
            match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', inner)
            if match:
                link_text, url = match.groups()
                rich_text.append({
                    "type": "text",
                    "text": {"content": link_text, "link": {"url": url}},
                    "annotations": {"bold": True}
                })
        # Bold text
        elif part.startswith('**') and part.endswith('**'):
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
    headline: str,
    month_name: str,
    article_count: int
) -> str:
    """
    Create Monthly Summary entry in Newsletter Pipeline.
    Puts the headline in Summary property and full content in page body.

    Returns page ID.
    """
    title = f"Monthly Summary - {month_name}"

    # Convert markdown to Notion blocks
    content_blocks = markdown_to_notion_blocks(summary_text)

    properties = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Type": {"select": {"name": "Monthly Summary"}},
        "Date Found": {"date": {"start": date.today().isoformat()}},
        "Summary": {"rich_text": [{"text": {"content": headline}}]},
    }

    # Notion API caps children at 100 per request — create with first batch,
    # then append remaining blocks in chunks.
    BATCH_SIZE = 100
    response = notion.pages.create(
        parent={"database_id": NEWSLETTER_PIPELINE_DB},
        icon={"type": "emoji", "emoji": "📣"},
        properties=properties,
        children=content_blocks[:BATCH_SIZE]
    )
    page_id = response["id"]

    for i in range(BATCH_SIZE, len(content_blocks), BATCH_SIZE):
        notion.blocks.children.append(
            block_id=page_id,
            children=content_blocks[i:i + BATCH_SIZE]
        )

    logger.info(f"Created Monthly Summary entry: {title}")
    return page_id, response["url"]


# ============================================================================
# SLACK INTEGRATION
# ============================================================================

def post_to_slack(
    headline: str,
    month_name: str,
    notion_url: str,
    article_count: int
) -> bool:
    """
    Post monthly summary notification to Slack.

    Uses SLACK_WEBHOOK_URL environment variable.
    Returns True if successful, False otherwise.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.info("SLACK_WEBHOOK_URL not set, skipping Slack notification")
        return False

    # Add Planning Center workspace prefix to Notion URL
    # API returns: https://www.notion.so/Page-Name-id
    # We need: https://www.notion.so/planningcenter/Page-Name-id
    if notion_url and "notion.so/" in notion_url and "/planningcenter/" not in notion_url:
        notion_url = notion_url.replace("notion.so/", "notion.so/planningcenter/")

    # Build Slack Block Kit message
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📣 Monthly Summary - {month_name}",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{headline}*"
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Based on {article_count} articles from the Newsletter Pipeline"
                }
            ]
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Read Full Summary",
                        "emoji": True
                    },
                    "url": notion_url,
                    "style": "primary"
                }
            ]
        }
    ]

    payload = {
        "blocks": blocks,
        "text": f"Monthly Summary - {month_name}: {headline}"  # Fallback for notifications
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logger.info("Posted to Slack successfully")
                return True
            else:
                logger.warning(f"Slack returned status {response.status}")
                return False
    except urllib.error.URLError as e:
        logger.error(f"Failed to post to Slack: {e}")
        return False


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

    # Query recent articles and community discussions
    logger.info("Querying Newsletter Pipeline for recent articles and community discussions...")
    articles = query_recent_articles(notion, period_start, period_end)

    # Break down by type
    regular_articles = [a for a in articles if a["type"] == "Article"]
    community_discussions = [a for a in articles if a["type"] == "Community Discussion"]

    logger.info(f"Found {len(regular_articles)} articles and {len(community_discussions)} community discussions for {month_name}")

    if not articles:
        logger.info("No content to analyze. Exiting.")
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
        logger.info(f"Would analyze {len(regular_articles)} articles and {len(community_discussions)} community discussions")
        logger.info(f"Total content: {len(articles)} entries")
        logger.info(f"Digest length: {len(article_digest)} characters")
        logger.info("\nSample of digest (first 2000 chars):")
        logger.info("-" * 40)
        print(article_digest[:2000])
        if len(article_digest) > 2000:
            logger.info(f"\n... and {len(article_digest) - 2000} more characters")
        return

    # Generate summary with Claude
    logger.info("Generating trend analysis with Claude Sonnet...")
    headline, summary = generate_monthly_summary(
        anthropic_client,
        article_digest,
        period_description,
        len(articles)
    )

    logger.info("Summary generated successfully")
    logger.info(f"Headline: {headline}")
    logger.info("-" * 40)
    print(summary)
    logger.info("-" * 40)

    # Create Notion entry
    logger.info("Creating Monthly Summary entry in Notion...")
    page_id, page_url = create_summary_entry(
        notion,
        summary,
        headline,
        month_name,
        len(articles)
    )

    logger.info(f"Created Notion page: {page_id}")

    # Post to Slack
    post_to_slack(
        headline=headline,
        month_name=month_name,
        notion_url=page_url,
        article_count=len(articles)
    )

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monthly trend analysis from Newsletter Pipeline")
    parser.add_argument("--days", type=int, default=None,
                        help="Override: look back N days instead of previous calendar month (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be analyzed without calling Claude or creating entries")

    args = parser.parse_args()

    main(days_back=args.days, dry_run=args.dry_run)
