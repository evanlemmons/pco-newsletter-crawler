#!/usr/bin/env python3
"""Create Circle Community entry in Notion Source Registry."""

import os
import sys
from notion_client import Client

# Database IDs
SOURCE_REGISTRY_DB = "43d593469bb8458c96ce927600514907"

def create_circle_source():
    """Create Circle Community entry in Source Registry."""
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        print("❌ NOTION_API_KEY not set")
        sys.exit(1)

    notion = Client(auth=api_key)

    properties = {
        "Name": {
            "title": [{"text": {"content": "Circle Community"}}]
        },
        "URL": {
            "url": "https://circle.so"
        },
        "Status": {
            "select": {"name": "Active"}
        },
        "Check Frequency": {
            "select": {"name": "Weekly"}
        },
        "Category": {
            "select": {"name": "Customer Insights"}
        },
        "Type": {
            "select": {"name": "Blog"}
        },
        "Priority": {
            "select": {"name": "High"}
        },
        "Why It Matters": {
            "rich_text": [{
                "text": {
                    "content": "Planning Center's private community where church admins, pastors, and staff discuss their experiences. Direct source of customer feedback, pain points, feature requests, and sentiment. Critical for understanding user needs and product opportunities."
                }
            }]
        },
        "Crawl Notes": {
            "rich_text": [{
                "text": {
                    "content": "Uses Circle Admin API for data extraction. Analyzes posts and comments for thematic patterns rather than individual article extraction."
                }
            }]
        }
    }

    try:
        response = notion.pages.create(
            parent={"database_id": SOURCE_REGISTRY_DB},
            icon={"type": "emoji", "emoji": "💬"},
            properties=properties
        )

        print("✅ Successfully created Circle Community entry")
        print(f"   Page ID: {response['id']}")
        print(f"   URL: {response['url']}")
        return True

    except Exception as e:
        print(f"❌ Failed to create entry: {e}")
        return False

if __name__ == "__main__":
    success = create_circle_source()
    sys.exit(0 if success else 1)
