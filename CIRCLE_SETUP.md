# Circle Community Integration - Setup Guide

## What Was Implemented

A complete Circle community analysis system that:
1. Fetches posts and comments from all Circle community spaces via API
2. Analyzes content with Claude Sonnet to identify recurring themes and pain points
3. Creates thematic summary entries in the Newsletter Pipeline
4. Runs weekly via GitHub Actions

## Files Created

### 1. `circle_analyzer.py`
Main Python script that:
- Connects to Circle API using Bearer token authentication
- Fetches all community spaces
- Retrieves posts and comments from the past 7 days
- Uses Claude Sonnet to analyze content for themes
- Creates Newsletter Pipeline entries for significant findings
- Updates Source Registry with crawl status

**Key features:**
- Supports `--dry-run` for testing without creating entries
- Supports `--days N` to customize lookback period
- Supports `--debug` for verbose logging
- Rate limiting to respect API limits
- Comprehensive error handling

### 2. `.github/workflows/circle-analyzer.yml`
GitHub Actions workflow that:
- Runs every Monday at 6:30 AM UTC (offset from blog crawler)
- Can be manually triggered with custom parameters
- Uploads logs as artifacts for debugging

### 3. `test_circle_api.py`
Testing script to verify Circle API authentication and connectivity.

## Setup Steps

### Step 1: Create Notion Source Registry Entry ✅

Add a new page to your [Source Registry](https://www.notion.so/planningcenter/43d59346-9bb8-458c-96ce-927600514907) with:

| Field | Value |
|-------|-------|
| **Name** | Circle Community |
| **URL** | https://circle.so |
| **Status** | Active |
| **Check Frequency** | Weekly |
| **Category** | Church Tech |
| **Type** | Blog *(or add "Community" type)* |
| **Priority** | High |
| **Why It Matters** | Planning Center's private community where church admins, pastors, and staff discuss their experiences. Direct source of customer feedback, pain points, feature requests, and sentiment. |
| **Crawl Notes** | Uses Circle Admin API. Analyzes posts and comments for thematic patterns. |

### Step 2: Add GitHub Secret ✅

1. Go to: https://github.com/evanlemmons/pco-newsletter-crawler/settings/secrets/actions
2. Click **New repository secret**
3. Name: `CIRCLE_API_KEY`
4. Value: `y3UBpZswwM4yPXMgLxP2GVmu6SW3Cbaj`
5. Click **Add secret**

### Step 3: Test via GitHub Actions (Required)

**IMPORTANT**: Testing must ONLY be done via GitHub Actions dry runs, not locally. Environment secrets and variables are configured in GitHub and should not be used in local environments.

### Step 4: Manual GitHub Actions Trigger

Test the workflow manually:

1. Go to: https://github.com/evanlemmons/pco-newsletter-crawler/actions/workflows/circle-analyzer.yml
2. Click **Run workflow**
3. Select options:
   - **dry_run**: `true` (for first test)
   - **days_back**: `3` (less data for testing)
4. Click **Run workflow**
5. Monitor the run and check logs

### Step 5: Review Output

After a successful run (with dry_run=false):

1. Check the [Newsletter Pipeline](https://www.notion.so/planningcenter/2efabbce69a280409309d052751eec14) for new entries
2. Entries will have:
   - **Title**: "Circle: [Theme Name]"
   - **Type**: Article
   - **Topic**: Church Tech or Product Management
   - **Summary**: Rich analysis with pain points, quotes, sentiment
   - **Status**: Unreviewed
   - **Icon**: 💬

3. Review theme quality:
   - Are the themes meaningful and actionable?
   - Do they represent genuine patterns (not one-offs)?
   - Are the pain points specific and useful?

## How It Works

### Data Flow

```
1. Fetch Circle Spaces (all active spaces)
     ↓
2. For each space:
   - Fetch posts from past 7 days
   - Fetch comments for each post
     ↓
3. Aggregate content by space
     ↓
4. Claude Sonnet Analysis:
   - Identify recurring themes
   - Extract pain points
   - Assess sentiment
   - Find notable quotes
     ↓
5. Create Newsletter Pipeline entries
   (one per significant theme, max 5)
     ↓
6. Update Source Registry status
```

### AI Analysis

The Claude Sonnet prompt asks for:
- **Recurring themes**: Topics mentioned by multiple people
- **Pain points**: Specific frustrations and challenges
- **Feature requests**: Product capabilities being asked for
- **Sentiment**: Overall emotional tone
- **Notable insights**: Quotes that reveal customer needs

**Quality threshold**: Only creates entries for themes that:
- Appear in multiple posts/comments
- Represent significant pain points or opportunities
- Have notable sentiment
- Contain actionable insights

### Output Format

Each theme entry includes:
- **Title**: Descriptive theme name
- **Description**: 2-3 sentences about the discussion
- **Pain Points**: Bullet list of specific challenges
- **Quotes**: Direct customer quotes
- **Sentiment**: Tone analysis
- **Recommendations**: Suggested actions for Planning Center
- **Source URLs**: Links to original Circle posts

## Monitoring & Maintenance

### Weekly Automated Runs

The workflow runs every Monday at 6:30 AM UTC automatically. No action needed.

### Checking Status

Use the GitHub MCP server or visit:
https://github.com/evanlemmons/pco-newsletter-crawler/actions/workflows/circle-analyzer.yml

### Common Issues

**No themes found:**
- This is normal if the community was quiet that week
- Check Circle directly to verify activity levels

**API errors:**
- Verify CIRCLE_API_KEY is valid and has correct permissions
- Check Circle API status/changes
- Review error logs in GitHub Actions artifacts

**Too many/few entries:**
- Adjust `MAX_THEMES_PER_RUN` constant in circle_analyzer.py (currently 5)
- Modify AI prompt to be more/less selective
- Change `days_back` parameter (default: 7)

### Adjusting Analysis

To tune the analysis quality, edit the `CIRCLE_ANALYSIS_PROMPT` in circle_analyzer.py:
- Make it more selective: emphasize "only actionable insights"
- Make it broader: remove some filtering criteria
- Change focus: add specific topics to look for

## Next Steps

Once you're comfortable with the Circle integration:

1. **Monitor output quality** for first few weeks
2. **Adjust prompt** if themes are too broad/narrow
3. **Consider space filtering** if certain spaces are noisy
4. **Add to monthly summary** to include Circle insights in trend reports
5. **Set up Slack notifications** for high-priority themes

## Future Enhancements

Potential improvements:
- Filter to specific high-value spaces only
- Track sentiment trends over time
- Identify most active/influential members
- Compare themes week-over-week
- Alert PM team on urgent pain points
- Generate weekly digest emails

## Support

If issues arise:
- Check GitHub Actions logs
- Run locally with `--debug` flag
- Review Crawl Notes in Source Registry
- Consult the main plan: `/Users/evanlemmons/.claude/plans/cozy-watching-pixel.md`
