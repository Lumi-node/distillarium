# Deployment — The Distillery

End-to-end steps to take the repo from local → public.

## Identifiers (not secrets — safe to share/commit)

- **Cloudflare Account ID:** `36ab6ec48b794825f43851bdc82f75a6`
- **Cloudflare Zone ID (thedistillery.run):** `6608086ff25ecf1466fbeedd702f1bf1`

## Secrets you need to create (NEVER commit these)

### Local (your machine — in `.env`)

| Secret | Already set? | Used by |
|---|---|---|
| `DISTILLARIUM_PYPI_KEY` | ✅ yes | `distillery_pkg/release.sh publish` |
| `TESTPYPI_API_KEY` | optional | `distillery_pkg/release.sh test` |
| `PYPI_API_KEY` | ✅ yes (legacy, account-wide) | (older fallback — prefer the project-scoped key above) |
| `GOOGLE_API_KEY` | ✅ yes | Gemini teacher for distillations |

### CI (GitHub repo Settings → Secrets → Actions)

| Secret | Used by |
|---|---|
| `CLOUDFLARE_API_TOKEN` | `.github/workflows/deploy-site.yml` for auto-deploys |
| `PYPI_API_TOKEN` | `.github/workflows/publish-pypi.yml` on tag push (paste the same value as `DISTILLARIUM_PYPI_KEY`) |
| `TEST_PYPI_API_TOKEN` | Optional — for `workflow_dispatch` test runs |

## Step 1 — Generate the Cloudflare API token

1. Go to https://dash.cloudflare.com/profile/api-tokens
2. Click **"Create Token"**
3. Use the **"Edit Cloudflare Workers"** template (it includes the right Pages permissions) OR create a custom token with these permissions:

   | Permission | Resource |
   |---|---|
   | Account → Cloudflare Pages → Edit | All accounts (or just the Distillery account) |
   | Account → Workers Scripts → Edit | All accounts (for future Workers) |
   | Account → Workers R2 Storage → Edit | All accounts (for Spirit downloads) |
   | Zone → DNS → Edit | `thedistillery.run` (+ `distillarium.app` once registered) |
   | Zone → Page Rules → Edit | `thedistillery.run` (for redirects) |

4. Copy the token (only shown once)
5. Add it to GitHub Secrets as `CLOUDFLARE_API_TOKEN`

## Step 2 — Create the Cloudflare Pages project

1. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**
2. Authorize GitHub, pick `the-distillery/distillarium`
3. Configure build:
   - **Build command:** `cd site && npm install && npm run build`
   - **Build output:** `site/dist`
   - **Root directory:** `/` (default)
   - **Project name:** `thedistillery`
4. Click **Save and Deploy** — first build kicks off
5. Once deployed, you'll get `thedistillery.pages.dev`

## Step 3 — Add custom domain(s)

1. In the Pages project → **Custom domains**
2. Add `distillarium.app` (after registering it) as primary
3. Add `thedistillery.run` as alias
4. Cloudflare auto-configures DNS for any domain on your Cloudflare account

## Step 4 — 301 redirect: `thedistillery.run` → `distillarium.app`

Two options:

**Option A — Cloudflare Bulk Redirects (recommended, free):**
- Dashboard → **Rules → Redirect Rules** for the `thedistillery.run` zone
- Source: `https://thedistillery.run/*`
- Destination: `https://distillarium.app/$1`
- Status: 301
- This preserves paths: `thedistillery.run/cellar/needle` → `distillarium.app/cellar/needle`

**Option B — keep both as live:**
- Both domains serve the same Cloudflare Pages build
- "The Distillery" brand keeps the `.run` domain operational
- Decide later if you want to consolidate

## Step 5 — Set up Cloudflare R2 for Spirit downloads

1. Dashboard → **R2** → **Create bucket** → name: `distillery-spirits`
2. Settings → **Public Access** → **Connect to public domain**:
   - Custom domain: `r2.distillarium.app`
   - Cloudflare auto-issues TLS cert + DNS
3. Upload Needle Spirit via dashboard or CLI:
   ```bash
   wrangler r2 object put distillery-spirits/needle-0.1.0.pt \
     --file=lab_builds/hn-48111896/checkpoints/needle_1k.pt
   ```
4. Update `site/src/data/cellar.json` — set `downloads.pytorch.url`:
   ```json
   "url": "https://r2.distillarium.app/needle-0.1.0.pt"
   ```

## Step 6 — Generate PyPI tokens for CI publishing

1. https://pypi.org/manage/account/token/ → **Add API token**
   - Token name: `distillarium-github-actions`
   - Scope: project `distillarium`
   - Copy the token (starts with `pypi-`)
   - Add to GitHub Secrets as `PYPI_API_TOKEN`
2. Same for https://test.pypi.org/manage/account/token/ → `TEST_PYPI_API_TOKEN`

## Step 7 — Push to GitHub

```bash
cd /media/lumi-node/Storage2/research-radar
git remote add origin git@github.com:the-distillery/distillarium.git
git push -u origin main
```

After push:
- `.github/workflows/deploy-site.yml` deploys the Astro site on every push to `site/**`
- `.github/workflows/publish-pypi.yml` publishes to PyPI on every tag matching `v*.*.*`

## Releasing a new version

### Option A — local (immediate, uses `DISTILLARIUM_PYPI_KEY` from `.env`)

```bash
cd distillery_pkg
# 1. Bump version in pyproject.toml (e.g. 0.1.0 → 0.1.1)
$EDITOR pyproject.toml

# 2. Sanity-check the build
./release.sh check

# 3. (optional) Dry-run on TestPyPI first
./release.sh test

# 4. Real publish — prompts for confirmation
./release.sh publish
```

### Option B — CI (push a git tag)

```bash
# Bump version in pyproject.toml, commit
git commit -am "bump distillarium to v0.1.1"

# Tag and push — triggers .github/workflows/publish-pypi.yml
git tag v0.1.1
git push origin main --tags
```

> **PyPI quirk:** a published version can be yanked but never republished. Always
> bump before re-uploading. `release.sh publish` will fail loudly if you try to
> overwrite an existing version.

## Quick local sanity checks before deploying

```bash
# Site builds clean
cd site && npm run build
ls dist/index.html dist/cellar/needle/index.html

# Package builds + checks clean
cd ../distillery_pkg && python -m build && twine check dist/*

# Tests pass
pytest tests/ -p no:anchorpy
```

## Cost ledger

- Cloudflare Pages: $0 (free tier)
- Cloudflare R2 storage: ~$0.50/mo (5 Spirits × 250 MB × $0.015/GB)
- Cloudflare R2 egress: **$0** (zero-egress is the whole point)
- Domain registration: ~$15-20/yr
- PyPI: $0
- GitHub public repo: $0
- **Total: ~$15-20/yr**
