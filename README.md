# Newsletter Pipeline Crawler

Automated web crawler that monitors sources defined in Notion and populates a Newsletter Pipeline with discovered articles.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions                            │
│                    (runs daily at 6 AM)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    newsletter_crawler.py                     │
├─────────────────────────────────────────────────────────────┤
│  1. Check today's date                                       │
│     → Daily sources: every day                               │
│     → Weekly sources: Mondays                                │
│     → Monthly sources: 1st of month                          │
│     → Quarterly sources: Jan 1, Apr 1, Jul 1, Oct 1          │
│                                                              │
│  2. Query Notion Source Registry                             │
│     → Get active sources matching today's frequencies        │
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
2. Or just wait for 6 AM UTC and it will run automatically

---

## Configuration

### Adjusting the Schedule

Edit `.github/workflows/newsletter-crawler.yml`:

```yaml
schedule:
  - cron: '0 6 * * *'  # 6 AM UTC daily
```

Cron format: `minute hour day month weekday`

Examples:
- `'0 11 * * *'` = 11 AM UTC (6 AM EST)
- `'0 14 * * *'` = 2 PM UTC (6 AM PST)
- `'0 6,18 * * *'` = 6 AM and 6 PM UTC

### Source Registry Fields

Configure these in Notion for each source:

| Field | Required | Description |
|-------|----------|-------------|
| Name | Yes | Source name for logging |
| URL | Yes | Base URL to crawl |
| Status | Yes | Must be "Active" to be crawled |
| Check Frequency | Yes | Daily, Weekly, Monthly, or Quarterly |
| Category | No | Maps to Topic in Pipeline |
| Crawl Pattern | No | URL filter, e.g., `*blog*,*news*` |
| Max Pages | No | Limit pages per crawl (default: 20) |
| Max Depth | No | Link depth to follow (default: 2) |

### Frequency Schedule

| Frequency | When It Runs |
|-----------|--------------|
| Daily | Every day |
| Weekly | Mondays |
| Monthly | 1st of each month |
| Quarterly | Jan 1, Apr 1, Jul 1, Oct 1 |

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

## Files

| File | Purpose |
|------|---------|
| `newsletter_crawler.py` | Main Python script |
| `requirements.txt` | Python dependencies |
| `.github/workflows/newsletter-crawler.yml` | GitHub Actions workflow |

---

## Cost

**Free!** GitHub Actions provides 2,000 minutes/month for free on private repos. A typical crawl run takes 5-15 minutes depending on source count.
