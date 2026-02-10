#!/usr/bin/env python3
"""Quick test script to verify Circle API authentication."""

import requests
import json
import sys

API_KEY = "y3UBpZswwM4yPXMgLxP2GVmu6SW3Cbaj"
BASE_URL = "https://app.circle.so/api/admin/v2"

def test_endpoint(endpoint_path, description):
    """Test a specific endpoint."""
    url = f"{BASE_URL}/{endpoint_path}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    params = {"per_page": 5}

    try:
        print(f"\nTrying {description}...")
        print(f"  URL: {url}")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        print(f"  ✓ Success! Found {len(data.get('records', []))} items")

        records = data.get("records", [])
        if records:
            print(f"  Sample item: {json.dumps(records[0], indent=2)[:200]}...")

        return True, data

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed: {e.response.status_code if hasattr(e, 'response') else e}")
        return False, None


def test_fetch_spaces():
    """Test various endpoints to find the right one."""
    print("Testing Circle API authentication...")
    print("=" * 60)

    # Try different endpoint variations
    endpoints = [
        ("community_spaces", "Community Spaces"),
        ("spaces", "Spaces"),
        ("comments/posts", "Posts (via comments)"),
        ("communities", "Communities"),
    ]

    for endpoint, description in endpoints:
        success, data = test_endpoint(endpoint, description)
        if success:
            return True

    print("\n✗ None of the tested endpoints worked")
    print("You may need to check the Circle API documentation for the correct endpoint")
    return False

if __name__ == "__main__":
    success = test_fetch_spaces()
    sys.exit(0 if success else 1)
