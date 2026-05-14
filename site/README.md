# thedistillery.run — site

Static site for The Distillery, built with Astro 5. Deploys to Cloudflare Pages.

## Local dev

```bash
cd site
npm install
npm run dev
# → http://localhost:4321
```

## Build

```bash
npm run build
npm run preview  # local preview of the build output
```

## Deploy to Cloudflare Pages

### One-time setup

1. Push this repo to GitHub
2. In Cloudflare dashboard → Pages → Create project → Connect to Git
3. Build command: `cd site && npm install && npm run build`
4. Build output directory: `site/dist`
5. Add custom domain: `thedistillery.run`

After that, every push to `main` triggers a deploy.

### Or via wrangler CLI

```bash
npx wrangler pages deploy ./dist --project-name thedistillery
```

## Structure

```
site/
├── astro.config.mjs
├── package.json
├── public/
│   ├── favicon.svg
│   ├── _headers       # Cloudflare Pages edge headers
│   └── robots.txt
├── src/
│   ├── layouts/
│   │   └── Base.astro
│   ├── components/
│   │   ├── BottleCard.astro
│   │   └── StillCanvas.astro    # the centerpiece animation
│   ├── data/
│   │   └── cellar.json          # public Spirits catalog
│   ├── pages/
│   │   ├── index.astro          # landing page
│   │   ├── docs.astro
│   │   └── cellar/
│   │       ├── index.astro      # all spirits
│   │       └── [slug].astro     # dynamic spirit detail (one per cellar.json entry)
│   └── styles/
│       └── global.css
└── wrangler.toml
```

## Adding a new Spirit

1. Add an entry to `src/data/cellar.json`
2. Rebuild — the `[slug].astro` route generates a detail page automatically

That's it. No DB needed.

## Cellar artifact hosting

Model `.pt` files live in Cloudflare R2. Update the `downloads.*.url` field in
`cellar.json` to point at the R2 public URL when a Spirit is ready to ship.

```
https://r2.thedistillery.run/spirits/needle-v1.pt
```
