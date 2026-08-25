# trading-bot-dashboard

The public face of the Gate 0 paper desk: the static site and the rendered
data feed it reads.

- `index.html` — landing page
- `dashboard.html`, `app.html` — the desk views
- `data/data.json` — equity curves, trades, drawdowns and win rates, served
  at `/api/data` (see `vercel.json`)

The engine, its `paper/` state and the cron that advances them live in a
private repo and push `data/data.json` here. Nothing in this repo executes.
