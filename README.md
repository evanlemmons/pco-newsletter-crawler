# Newsletter Pipeline Crawler

Automated web crawler that monitors sources defined in Notion and populates a Newsletter Pipeline with discovered articles.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions                            │
│                    (runs weekly on Mondays at 6 AM UTC)      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    newsletter_crawler.py                     │
├─────────────────────────────────────────────────────────────┤
│  1. Query Notion Source Registry for active sources          │
│     → All "Weekly" frequency sources run every Monday        │
│     → "Monthly" sources run if 1st falls on Monday           │
│     → "Quarterly" sources run if quarter start is Monday     │
│                                                              │
│  2. For each matching source:                                │
│                                                              │
│  3. For each source:                                         │
│     → Crawl website using Crawl4AI                           │
│     → Extract articles (title, URL, summary)                 │
│     → Check Newsletter Pipeline for duplicates               │
│     → Create new entries for new articles                    │
│     → Update source status (Last Success, failures, etc.)    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         Notion                               │
├─────────────────────────────────────────────────────────────┤
│  📡 Source Registry          🗞️ Newsletter Pipeline          │
│  ─────────────────          ───────────────────────          │
│  • Configure sources        • New articles appear here       │
│  • Set frequencies          • Status: "Unreviewed"           │
│  • Monitor health           • Ready for your review          │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### Step 1: Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it something like `newsletter-crawler`
3. Make it **Private** (recommended since it will have your Notion data)
4. Click **Create repository**

### Step 2: Create a Notion Integration

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Click **+ New integration**
3. Name it `Newsletter Crawler`
4. Select your workspace
5. Click **Submit**
6. Copy the **Internal Integration Secret** (starts with `ntn_` or `secret_`)
7. **Important**: Keep this secret safe!

### Step 3: Connect Notion Databases to Integration

1. Open your **Source Registry** database in Notion
2. Click the `•••` menu in the top right
3. Click **Connections** → **Connect to** → Select **Newsletter Crawler**
4. Repeat for your **Newsletter Pipeline** database

### Step 4: Add Secret to GitHub

1. In your GitHub repository, go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `NOTION_API_KEY`
4. Value: Paste your Notion integration secret
5. Click **Add secret**

### Step 5: Upload the Files

Upload these files to your repository:

```
your-repo/
├── .github/
│   └── workflows/
│       └── newsletter-crawler.yml
├── newsletter_crawler.py
├── requirements.txt
└── README.md
```

**Option A: Using GitHub Web Interface**

1. In your repo, click **Add file** → **Upload files**
2. Drag and drop the files
3. For the `.github/workflows/` folder, you may need to create the folders first:
   - Click **Add file** → **Create new file**
   - Type `.github/workflows/newsletter-crawler.yml` as the filename
   - Paste the workflow content
   - Click **Commit new file**

**Option B: Using Git Command Line**

```bash
git clone https://github.com/YOUR_USERNAME/newsletter-crawler.git
cd newsletter-crawler

# Copy the files into the repo
# Then:
git add .
git commit -m "Initial setup"
git push
```

### Step 6: Test It

1. Go to **Actions** tab in your repository
2. Click **Newsletter Crawler** in the left sidebar
3. Click **Run workflow** dropdown (right side)
4. Check **Dry run** to test without creating entries
5. Click **Run workflow**
6. Watch the logs to see it query your Notion sources!

### Step 7: Run for Real

1. Same as above, but leave **Dry run** unchecked
2. Or wait for the next Monday at 6 AM UTC when it runs automatically

---

## Configuration

### Adjusting the Schedule

Edit `.github/workflows/newsletter-crawler.yml`:

```yaml
schedule:
  - cron: '0 6 * * 1'  # 6 AM UTC every Monday
```

Cron format: `minute hour day month weekday` (0=Sunday, 1=Monday, etc.)

Examples:
- `'0 6 * * 1'` = 6 AM UTC every Monday (current setting)
- `'0 6 * * *'` = 6 AM UTC daily
- `'0 11 * * 1'` = 11 AM UTC every Monday (6 AM EST)
- `'0 6 * * 1,4'` = 6 AM UTC on Mondays and Thursdays

### Source Registry Fields

Configure these in Notion for each source:

| Field | Required | Description |
|-------|----------|-------------|
| Name | Yes | Source name for logging |
| URL | Yes | Base URL to crawl |
| Status | Yes | Must be "Active" to be crawled |
| Check Frequency | Yes | Weekly (recommended), Monthly, or Quarterly |
| Category | No | Maps to Topic in Pipeline |
| Crawl Pattern | No | URL filter, e.g., `*blog*,*news*` |
| Max Pages | No | Limit pages per crawl (default: 20) |
| Max Depth | No | Link depth to follow (default: 2) |

### Frequency Schedule

The workflow runs **every Monday at 6 AM UTC**. Sources are included based on their frequency:

| Frequency | When It Runs |
|-----------|--------------|
| Weekly | Every Monday (recommended) |
| Monthly | When the 1st of month falls on a Monday |
| Quarterly | When Jan 1, Apr 1, Jul 1, or Oct 1 falls on Monday |

> **Note**: Since the workflow only runs on Mondays, "Weekly" is the most reliable frequency setting.

---

## Manual Commands

### Force All Sources

Run all active sources regardless of their frequency:

1. Actions → Newsletter Crawler → Run workflow
2. Check **Force all sources**
3. Run

### Dry Run

See what would be crawled without making changes:

1. Actions → Newsletter Crawler → Run workflow
2. Check **Dry run**
3. Run

---

## Troubleshooting

### "Crawl4AI not installed"
The workflow should install it automatically. Check the Actions logs for errors.

### "NOTION_API_KEY not set"
Make sure you added the secret in Settings → Secrets → Actions.

### "Source not being crawled"
Check that:
- Status is "Active"
- Check Frequency matches today (e.g., Weekly only runs on Mondays)
- Database is connected to the Notion integration

### "No articles found"
- Check your Crawl Pattern isn't too restrictive
- Try increasing Max Pages and Max Depth
- Some sites may block automated crawling

### Workflow not running
- Check Actions are enabled for your repository
- Scheduled workflows may be disabled after 60 days of repo inactivity

---

## GitHub MCP Server Integration

This project is configured to work with the [official GitHub MCP server](https://github.com/github/github-mcp-server), which provides direct API access to GitHub from Claude Code.

### Benefits of Using the MCP Server

- **Automated workflow management**: Trigger workflows, check run status, and view logs directly from Claude
- **No manual navigation**: Access GitHub Actions, issues, PRs, and files without opening a browser
- **Structured responses**: Get parsed data instead of HTML pages
- **Better automation**: Integrate GitHub operations into broader workflows

### Available Operations

The GitHub MCP server provides access to:
- Repository files and structure
- Workflow runs, logs, and manual triggers
- Issues and pull requests
- Commit history and branches
- Actions status and artifacts

### Setup

1. Install the GitHub MCP server: https://github.com/github/github-mcp-server
2. Add your GitHub Personal Access Token to the configuration
3. Configure in `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "/path/to/github-mcp-server",
      "args": ["stdio"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
      }
    }
  }
}
```

4. Enable the toolsets you need (repos, actions, issues, pull_requests, etc.)

See the [GitHub MCP server documentation](https://github.com/github/github-mcp-server) for detailed installation instructions.

---

## Files

| File | Purpose |
|------|---------|
| `newsletter_crawler.py` | Main Python script |
| `monthly_summary.py` | Monthly summary generator |
| `requirements.txt` | Python dependencies |
| `.github/workflows/newsletter-crawler.yml` | Weekly crawler workflow |
| `.github/workflows/monthly-summary.yml` | Monthly summary workflow |

---

## Cost

**Free!** GitHub Actions provides 2,000 minutes/month for free on private repos. A typical crawl run takes 5-15 minutes depending on source count.
