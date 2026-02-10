# Circle Integration - Ready for Testing

## ✅ Completed Setup

### 1. Notion Source Registry Entry Created
- **Entry**: Circle Community
- **Category**: Customer Insights (your new category!)
- **Status**: Active
- **Check Frequency**: Weekly
- **Page ID**: 303abbce-69a2-814d-97ac-c66ac2aa206e
- **View**: https://www.notion.so/Circle-Community-303abbce69a2814d97acc66ac2aa206e

### 2. GitHub Secret Added
- **Secret Name**: `CIRCLE_API_KEY`
- **Added**: 2026-02-10 at 20:38 UTC
- **Status**: ✅ Verified in repository secrets

### 3. Documentation Updated
- **[CLAUDE.md](CLAUDE.md)**:
  - Added Circle Community Analyzer to workflows section
  - Updated testing guidance to mandate GitHub Actions dry runs only
  - Removed local testing instructions

- **[CIRCLE_SETUP.md](CIRCLE_SETUP.md)**:
  - Removed local testing steps
  - Emphasized GitHub Actions-only testing policy

### 4. Enhanced Analyzer Features

#### Customer Quotes Extraction
The analyzer now:
- Extracts multiple impactful customer quotes (up to 3 per theme)
- Prioritizes emotional, specific quotes that illustrate pain/needs
- Includes author names when available
- Formats quotes prominently in summaries with 💬 emoji

#### High-Impact Thread Detection
The analyzer now:
- Identifies threads with high comment counts (5+ = high engagement)
- Prioritizes threads where customers are clearly passionate
- Separates high-impact threads from additional links
- Lists high-engagement threads first in summaries with 🔗 emoji

#### Enhanced AI Prompt
Updated to:
- Request actual customer quotes (not paraphrased)
- Focus on posts with many comments
- Separate high-impact threads from additional links
- Include thread engagement notes

#### Enhanced Summary Format
Newsletter Pipeline entries now include:
- **Space name** at the top
- **🔥 Thread Activity**: Notes on high-engagement threads
- **💔 Pain Points**: Bullet list of specific challenges
- **💬 Customer Voices**: Multiple customer quotes
- **😊 Sentiment**: Overall tone analysis
- **🔗 High-Engagement Threads**: Links to most active discussions
- **💡 Recommended Actions**: Specific suggestions

## 📋 What to Expect in Dry Run

When you approve the dry run, you'll see:

1. **Spaces Fetched**: List of all Circle community spaces
2. **Posts Retrieved**: Posts from past 7 days per space
3. **AI Analysis**: Claude Sonnet analyzing content for themes
4. **Sample Output**: Preview of what would be created

### Expected Log Output

```
============================================================
Circle Community Analyzer
============================================================
Analyzing content from past 7 days (since 2026-02-03)
Querying Source Registry for Circle Community...
Found Circle source: Circle Community

Fetching Circle community spaces...
Found 15 spaces

--- DRY RUN MODE ---
Would analyze space: Ask the Community (42 total posts)
Would analyze space: Feature Requests (18 total posts)
Would analyze space: General Discussion (31 total posts)
...
```

### Sample Theme Entry (What Would Be Created)

**Title**: Circle: Mobile App Performance Issues

**Summary**:
Space: Ask the Community

Multiple churches reporting slow performance with the mobile check-in app during peak Sunday morning times. Issues seem worse on older Android devices and when attendance is high.

🔥 Thread Activity: One thread has 12 comments from different churches all experiencing similar problems

💔 Pain Points:
• Check-in times taking 30+ seconds on busy Sundays
• App crashes when processing large family check-ins
• Older Android devices (3+ years) nearly unusable

💬 Customer Voices:
"We had families standing in line for 10 minutes last Sunday because the app kept timing out. People were frustrated and some just left." - Sarah M.

"Our children's ministry is considering going back to paper check-in. The app is too slow and unreliable during our busiest times." - Mike R.

"I love Planning Center but the mobile app performance is really hurting our Sunday morning experience." - Jennifer T.

😊 Sentiment: Negative/Frustrated - Churches love the product but are genuinely struggling with this issue

🔗 High-Engagement Threads:
• https://circle.so/c/ask-a-question/mobile-checkin-slow-sundays-12345
• https://circle.so/c/ask-a-question/android-app-crashes-6789

💡 Recommended Actions:
Investigate mobile app performance under high load, especially for Android devices. Consider load testing during peak usage patterns and optimizing check-in flow for slower devices.

**URL**: https://circle.so/c/ask-a-question/mobile-checkin-slow-sundays-12345
**Topic**: Church Tech
**Status**: Unreviewed

## 🚀 Ready to Test

Everything is configured and ready. When you're ready:

### Manual Dry Run Trigger

1. Go to: https://github.com/evanlemmons/pco-newsletter-crawler/actions/workflows/circle-analyzer.yml

2. Click **Run workflow**

3. Configure inputs:
   - **days_back**: `3` (recommended for first test - less data)
   - **dry_run**: `true` (test without creating entries)

4. Click **Run workflow**

5. Monitor the run at: https://github.com/evanlemmons/pco-newsletter-crawler/actions

6. Review output logs to see:
   - Which spaces were analyzed
   - How many posts/comments were found
   - What themes were identified
   - Sample summaries with quotes and links

### After Dry Run Review

If the output looks good:
1. Run again with `dry_run=false` to create actual entries
2. Check Newsletter Pipeline for new Circle entries
3. Review theme quality and quote selection
4. Adjust AI prompt if needed (see below)

## 🎛️ Tuning the Analyzer

If themes are too broad/narrow or quotes aren't impactful enough, you can adjust the AI prompt in [circle_analyzer.py](circle_analyzer.py):

- **Line 397**: `CIRCLE_ANALYSIS_PROMPT` - the main prompt
- **Line 419**: Quote requirements section
- **Line 425**: Thread engagement criteria

Common adjustments:
- Increase minimum comment threshold (currently 5+)
- Request more/fewer quotes per theme
- Change tone of quotes (more emotional, more specific, etc.)
- Adjust max themes per run (currently 5)

## 📊 Success Metrics

A successful Circle integration should:
- Capture 2-5 significant themes per week
- Include impactful customer quotes that make PMs take notice
- Link to high-engagement threads (5+ comments)
- Provide actionable insights, not just information
- Focus on patterns, not one-off complaints

## 🔄 Weekly Automation

Once tested and approved, the analyzer will run automatically:
- **Every Monday at 6:30 AM UTC**
- Analyzes past 7 days of Circle activity
- Creates Newsletter Pipeline entries for themes
- Updates Source Registry with status

No manual intervention needed after initial approval!
