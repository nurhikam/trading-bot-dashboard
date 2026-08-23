"""
sync.py — export paper/ CSVs to data.json and push to the dashboard GitHub repo.

  python3 sync.py          export data.json + git push
  python3 sync.py --push   same (explicit)
  python3 sync.py --demo   self-checks

The repo checkout lives in _data_repo/ (gitignored). Uses the same metrics
logic as dashboard.py so the JSON served to the live site matches the local
view. Best run from within multi.py after each cycle (see --sync there).
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from multi import SUMMARY_CSV, TRADES_CSV, build_config, max_drawdown_pct, read_wallet

DATA_REPO_URL = "https://github.com/nurhikam/trading-bot-dashboard.git"
DATA_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data_repo")
DATA_JSON = os.path.join(DATA_REPO_DIR, "data", "data.json")


def compute_metrics(summary, trades, strategies, balance, symbol_hint=None):
    metrics = {}
    for s in strategies:
        col = f"eq_{s}"
        if col not in summary.columns:
            continue
        eq = summary[col].astype(float).dropna()
        if len(eq) == 0:
            continue
        last = float(eq.iloc[-1])
        pnl = (last / balance - 1) * 100
        dd = max_drawdown_pct(eq.tolist())

        st = trades[trades["strategy"] == s] if trades is not None and len(trades) else pd.DataFrame()
        n = len(st)
        fees = float(st["fee"].sum()) if n else 0.0

        wins, closes, long = 0, 0, 0.0
        for _, t in st.iterrows():
            if t["side"] == "BUY":
                long = float(t["price"])
            elif long:
                closes += 1
                wins += 1 if float(t["price"]) > long else 0
                long = 0.0
        win_rate = wins / closes * 100 if closes else 0.0

        in_col = f"in_{s}"
        in_pos = bool(summary[in_col].iloc[-1]) if in_col in summary.columns and len(summary) else False

        metrics[s] = {
            "pnl": round(pnl, 3), "dd": round(dd, 3), "trades": n,
            "fees": round(fees, 3), "win_rate": round(win_rate, 1),
            "last": round(last, 3), "in_pos": in_pos,
            "halted": bool(read_wallet(s, balance, symbol_hint).get("halted")),
        }
    return metrics


def build_data_json(cfg, live=None):
    symbols = cfg.get("symbols") or ([cfg.get("symbol")] if cfg.get("symbol") else ["BTC/USDT"])
    coins = {}
    for symbol in symbols:
        sp = os.path.join(os.path.dirname(SUMMARY_CSV), f"summary_{symbol.replace('/', '_')}.csv")
        tp = os.path.join(os.path.dirname(TRADES_CSV), f"trades_{symbol.replace('/', '_')}.csv")
        # fallback to legacy single file for first symbol
        if not os.path.exists(sp):
            sp = SUMMARY_CSV
            tp = TRADES_CSV
        if not os.path.exists(sp):
            continue
        summary = pd.read_csv(sp)
        try:
            trades = pd.read_csv(tp, on_bad_lines="skip", engine="python") if os.path.exists(tp) else pd.DataFrame()
        except Exception:
            trades = pd.DataFrame()
        strategies = [s for s in cfg["strategies"] if f"eq_{s}" in summary.columns]
        metrics = compute_metrics(summary, trades, strategies, cfg["balance"], symbol_hint=symbol)
        MAX_ROWS = 500  # prune payload; CSVs in repo keep full history
        summary_trim = summary.tail(MAX_ROWS)
        trades_trim = trades.tail(MAX_ROWS * 4) if len(trades) else trades
        coins[symbol] = {
            "symbol": symbol,
            "first_ts": str(summary["ts_iso"].iloc[0]) if len(summary) else None,
            "last_ts": str(summary["ts_iso"].iloc[-1]) if len(summary) else None,
            "n_candles": len(summary),
            "buy_hold": round((float(summary["close"].iloc[-1]) / float(summary["close"].iloc[0]) - 1) * 100, 3) if len(summary) else 0.0,
            "last_close": round(float(summary["close"].iloc[-1]), 2) if len(summary) else 0.0,
            "summary": summary_trim.to_dict(orient="records"),
            "trades": trades_trim.to_dict(orient="records") if len(trades_trim) else [],
            "metrics": metrics,
        }
    if not coins:
        return None
    # top-level for backward compat = first symbol
    first = coins[symbols[0]] if symbols[0] in coins else next(iter(coins.values()))
    doc = {
        "symbols": list(coins.keys()),
        "symbol": first["symbol"],
        "timeframe": cfg["timeframe"],
        "balance": cfg["balance"],
        "strategies": cfg["strategies"],
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_ts": first["first_ts"],
        "last_ts": first["last_ts"],
        "n_candles": first["n_candles"],
        "buy_hold": first["buy_hold"],
        "last_close": first["last_close"],
        "summary": first["summary"],
        "trades": first["trades"],
        "metrics": first["metrics"],
        "coins": coins,
    }
    if live:
        doc["live"] = live
    return doc


def export(cfg, live=None):
    """Write data.json; returns True if written, False on no-data, None if unchanged."""
    doc = build_data_json(cfg, live=live)
    if doc is None:
        print("No paper data yet — run multi.py first.")
        return False
    os.makedirs(os.path.dirname(DATA_JSON), exist_ok=True)
    payload = json.dumps(doc, indent=2)

    if os.path.exists(DATA_JSON):
        try:
            with open(DATA_JSON) as f:
                if f.read() == payload:
                    return None  # nothing changed — skip push
        except OSError:
            pass

    with open(DATA_JSON, "w") as f:
        f.write(payload)
    print(f"Exported {len(payload):,} bytes -> {os.path.relpath(DATA_JSON)}")
    return True


def reconcile():
    """Sync local clone to origin/main BEFORE exporting.

    data.json is fully regenerated from paper/ CSVs every cycle, so we can
    always hard-take remote: design commits land there, and any local-only
    state is disposable. This kills rebase conflicts permanently.
    """
    r = subprocess.run(["git", "-C", DATA_REPO_DIR, "fetch", "origin", "main", "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SYNC fetch failed\n{r.stderr}")
        return False
    r = subprocess.run(["git", "-C", DATA_REPO_DIR, "rev-parse", "HEAD", "origin/main"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SYNC rev-parse failed\n{r.stderr}")
        return False
    head, remote = r.stdout.split()
    if head != remote:
        rr = subprocess.run(["git", "-C", DATA_REPO_DIR, "reset", "--hard", "origin/main", "--quiet"],
                            capture_output=True, text=True)
        if rr.returncode != 0:
            print(f"SYNC reset failed\n{rr.stderr}")
            return False
    return True


def commit_push(commit_msg=None):
    """Commit exported data and push. Amend only when HEAD is our own data commit."""
    msg = commit_msg or f"data: update {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    r = subprocess.run(["git", "-C", DATA_REPO_DIR, "log", "-1", "--pretty=%s"],
                       capture_output=True, text=True)
    last_is_data = r.stdout.strip().startswith("data:") if r.returncode == 0 else False
    commit_cmd = (["git", "-C", DATA_REPO_DIR, "commit", "--amend", "-m", msg, "--quiet"]
                  if last_is_data else
                  ["git", "-C", DATA_REPO_DIR, "commit", "-m", msg, "--quiet"])
    cmds = [
        ["git", "-C", DATA_REPO_DIR, "add", "-A"],
        commit_cmd,
        ["git", "-C", DATA_REPO_DIR, "push", "origin", "main", "--quiet"],
    ]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode not in (0, 1):  # 1 = nothing to commit, fine
            print(f"SYNC ERROR: {' '.join(c)}\n{r.stderr}")
            return False
    return True


def sync(cfg, commit_msg=None, live=None):
    """reconcile -> export -> push. Returns True on success/nothing-new."""
    if not setup_sync():
        return False
    if not reconcile():
        return False
    res = export(cfg, live=live)
    if res is None:
        return True  # nothing changed since last publish
    if not res:
        return False
    return commit_push(commit_msg)


def setup_sync():
    """Clone the data repo into _data_repo/ if missing."""
    if not os.path.exists(os.path.join(DATA_REPO_DIR, ".git")):
        os.makedirs(DATA_REPO_DIR, exist_ok=True)
        r = subprocess.run(["git", "clone", DATA_REPO_URL, DATA_REPO_DIR],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"SYNC ERROR cloning: {r.stderr}")
            return False
    return True


def demo():
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    orig_summary, orig_trades = globals().get("SUMMARY_CSV"), globals().get("TRADES_CSV")
    orig_data = globals().get("DATA_JSON")
    globals()["SUMMARY_CSV"] = os.path.join(tmp, "summary.csv")
    globals()["TRADES_CSV"] = os.path.join(tmp, "trades.csv")
    globals()["DATA_JSON"] = os.path.join(tmp, "data.json")

    rows = []
    for i in range(5):
        rows.append({"ts_iso": f"2026-08-22T1{i}:00:00+00:00", "ts_ms": 1000 + i, "close": 100 + i * 10,
                     "eq_supertrend": 1000 + i * 5, "in_supertrend": 1})
    pd.DataFrame(rows).to_csv(globals()["SUMMARY_CSV"], index=False)
    pd.DataFrame([
        {"ts_iso": "x", "ts_ms": 1, "strategy": "supertrend", "side": "BUY", "price": 100, "qty": 0.01, "notional": 1, "fee": 0.1},
        {"ts_iso": "x", "ts_ms": 2, "strategy": "supertrend", "side": "SELL", "price": 150, "qty": 0.01, "notional": 1.5, "fee": 0.1},
    ]).to_csv(globals()["TRADES_CSV"], index=False)

    cfg = {"symbol": "BTC/USDT", "timeframe": "1h", "strategies": ["supertrend"], "balance": 1000.0}
    doc = build_data_json(cfg)
    assert doc is not None
    assert doc["n_candles"] == 5, doc["n_candles"]
    assert doc["metrics"]["supertrend"]["trades"] == 2
    assert doc["metrics"]["supertrend"]["win_rate"] == 100.0

    try:
        globals()["SUMMARY_CSV"], globals()["TRADES_CSV"] = orig_summary, orig_trades
        if orig_data is not None:
            globals()["DATA_JSON"] = orig_data
    finally:
        shutil.rmtree(tmp)
    print("demo() self-checks passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        setup_sync()
        sync(build_config())
