"""
CI runner — one desk pass for GitHub Actions.

Reads committed paper/ state, processes any newly closed candles across all
symbols, refreshes live price tick, and rewrites data/data.json. Git commit
and push are handled by the workflow steps around this script.
"""
import json
import os
import sys

os.environ.setdefault(
    "SYMBOLS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,PEPE/USDT,SUI/USDT,ADA/USDT,LINK/USDT,SHIB/USDT,XLM/USDT,LTC/USDT,UNI/USDT,AVAX/USDT,NEAR/USDT,BCH/USDT,HBAR/USDT,FET/USDT,DOT/USDT,FIL/USDT,ETC/USDT,APT/USDT,INJ/USDT,SEI/USDT"
)
os.environ.setdefault("TIMEFRAME", "1h")
os.environ.setdefault("BINANCE_HOST", "data-api.binance.vision")  # public mirror, dodges 451 geo-block on cloud IPs

from multi import build_config, build_exchange, step, read_wallet, live_tick  # noqa: E402


NOTIFY_STATE = os.path.join("paper", "notify_state.json")


def tg_send(text):
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False
    import urllib.request
    import urllib.parse
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print(f"telegram send failed: {e}")
        return False


def wib_stamp(iso_or_ms):
    from datetime import datetime, timezone, timedelta
    if isinstance(iso_or_ms, (int, float)):
        dt = datetime.fromtimestamp(iso_or_ms / 1000, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(iso_or_ms).replace("Z", "+00:00"))
    return (dt + timedelta(hours=7)).strftime("%d %b %H:%M WIB")


def collect_new_fills(cfg):
    """Return (fills grouped by symbol, max ts_ms seen) for trades newer than notify cursor."""
    import glob
    import csv
    state_path = NOTIFY_STATE
    last_ts = 0
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                last_ts = int(json.load(f).get("last_ts", 0))
        except Exception:
            last_ts = 0
    fills = {}
    max_seen = last_ts
    for sym in cfg["symbols"]:
        slug = sym.replace("/", "_")
        fp = os.path.join("paper", f"trades_{slug}.csv")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for row in csv.DictReader(f):
                ts = int(float(row["ts_ms"]))
                max_seen = max(max_seen, ts)
                if ts > last_ts:
                    fills.setdefault(sym, []).append(row)
    return fills, max_seen, state_path


def main():
    cfg = build_config()
    exchange = build_exchange()
    errors = 0
    for sym in cfg["symbols"]:
        try:
            wallets = {s: read_wallet(s, cfg["balance"], sym) for s in cfg["strategies"]}
            step(exchange, cfg, sym, wallets)
        except Exception as e:
            errors += 1
            print(f"{sym} step failed: {e}")

    live = None
    try:
        live = live_tick(exchange, cfg)
    except Exception as e:
        print(f"live tick failed: {e}")

    from sync import build_data_json
    doc = build_data_json(cfg, live=live)
    if doc is None:
        print("no data produced")
        sys.exit(1)

    # --- telegram: new fills ---
    try:
        fills, max_seen, nspath = collect_new_fills(cfg)
        if fills:
            lines = []
            n = 0
            for sym in cfg["symbols"]:
                for row in fills.get(sym, []):
                    arrow = "\U0001F7E2" if row["side"] == "BUY" else "\U0001F534"
                    px = float(row["price"])
                    lines.append(f'{arrow} {sym} <b>{row["strategy"]}</b> '
                                 f'{row["side"]} @ {px:,.4g} · {wib_stamp(int(row["ts_ms"]))}')
                    n += 1
                    if n >= 25:
                        break
                if n >= 25:
                    break
            if n:
                tg_send("<b>Paper desk fills</b>\n" + "\n".join(lines))
        with open(nspath, "w") as f:
            json.dump({"last_ts": max_seen}, f)
    except Exception as e:
        print(f"notify error: {e}")

    # --- telegram: 07:00 WIB daily digest ---
    try:
        from datetime import datetime, timezone, timedelta
        now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
        marker = os.path.join("paper", ".digest_sent")
        sent_on = ""
        if os.path.exists(marker):
            with open(marker) as f:
                sent_on = f.read().strip()
        today = now_wib.strftime("%Y-%m-%d")
        if now_wib.hour == 7 and sent_on != today and int(now_wib.minute) < 10:
            lines = ["<b>Morning digest — paper desk</b>"]
            for sym in cfg["symbols"]:
                sp = os.path.join("paper", f"summary_{sym.replace('/', '_')}.csv")
                if not os.path.exists(sp):
                    continue
                import csv as _csv
                with open(sp) as f:
                    rows = list(_csv.DictReader(f))
                if len(rows) < 2:
                    continue
                last_r, day_ago = rows[-1], rows[max(0, len(rows) - 24)]
                deltas = []
                for s in cfg["strategies"]:
                    a = float(day_ago[f"eq_{s}"]); b = float(last_r[f"eq_{s}"])
                    deltas.append((b / a - 1) * 100 if a else 0.0)
                best_s = cfg["strategies"][max(range(len(deltas)), key=lambda i: deltas[i])]
                bh_a = float(day_ago["close"]); bh_b = float(last_r["close"])
                bh_d = (bh_b / bh_a - 1) * 100 if bh_a else 0.0
                lines.append(f'{sym}: best <b>{best_s} {deltas[max(deltas.index(max(deltas)))]:+.2f}%</b>'
                             f' vs B&amp;H {bh_d:+.2f}%')
            tg_send("\n".join(lines))
            with open(marker, "w") as f:
                f.write(today)
    except Exception as e:
        print(f"digest error: {e}")

    with open("data/data.json", "w") as f:
        json.dump(doc, f, indent=2)

    if errors == len(cfg["symbols"]):
        print("all symbols failed — failing the run")
        sys.exit(1)

    print(json.dumps({
        "coins": len(doc["coins"]),
        "candles": doc["n_candles"],
        "updated": doc["updated_utc"],
        "live": bool(live),
        "errors": errors,
    }))


if __name__ == "__main__":
    main()
