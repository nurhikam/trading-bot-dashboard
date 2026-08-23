"""
Multi-strategy paper runner — N strategies on the same market data, each with
its own virtual wallet, everything logged for later comparison.

  python3 multi.py --demo      self-checks
  python3 multi.py --once      run one cycle then exit
  python3 multi.py             run forever, waking after each candle close
  python3 multi.py --report    print comparison table from logged history
  python3 multi.py --reset     wipe logs & states for a fresh start

Market data is Binance mainnet public OHLCV (no keys needed), so the
comparison is against the real market. Each strategy has its own virtual
wallet, so they never compete for the same funds.

Logs written under paper/ (gitignored):
  summary.csv    one row per candle: ts, close, equity & position per strategy
  trades.csv     every executed trade across all strategies
  paper.log      human-readable cycle log
  state_<s>.json restart-safe virtual wallet per strategy

Env:
  SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT
  SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT
  SYMBOL=BTC/USDT  # legacy single-symbol fallback  # legacy single-symbol fallback
  TIMEFRAME=1h
  STRATEGIES=supertrend,bb_squeeze,sma_trend,sma_cross,donchian,trend_200,rsi_revert
  START_BALANCE=1000
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

import bot
from live import load_env

PAPER_DIR = "paper"
SUMMARY_CSV = os.path.join(PAPER_DIR, "summary.csv")
TRADES_CSV = os.path.join(PAPER_DIR, "trades.csv")
PAPER_LOG = os.path.join(PAPER_DIR, "paper.log")

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
DEFAULT_STRATEGIES = ["supertrend", "bb_squeeze", "sma_trend",
                      "sma_cross", "donchian", "trend_200", "rsi_revert",
                      "macd", "keltner", "pctb_revert", "roc_mom", "obv_slope"]


# --- helpers ------------------------------------------------------------

def log(msg):
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(PAPER_LOG, "a") as f:
        f.write(line + "\n")


def build_config():
    load_env()
    sym_raw = os.environ.get("SYMBOLS") or os.environ.get("SYMBOL", "BTC/USDT")
    symbols = [s.strip() for s in sym_raw.split(",") if s.strip()]
    strat = os.environ.get("STRATEGIES", ",".join(DEFAULT_STRATEGIES))
    strategies = [s.strip() for s in strat.split(",") if s.strip()]
    for s in strategies:
        if s not in bot.STRATEGIES:
            sys.exit(f"Unknown STRATEGY={s}, pick from {list(bot.STRATEGIES)}")
    cfg = {
        "symbols": symbols,
        "symbol": symbols[0],
        "timeframe": os.environ.get("TIMEFRAME", "1h"),
        "strategies": strategies,
        "balance": float(os.environ.get("START_BALANCE", 1000)),
    }
    return cfg


def build_exchange():
    # spot-only: paper desk trades spot, and futures endpoints have no geo-block-free mirror
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"fetchMarkets": ["spot"]},
    })
    host = os.environ.get("BINANCE_HOST")
    if host:
        # remap every api.binance.com URL to the mirror (version-agnostic)
        for section, url in list(ex.urls["api"].items()):
            if isinstance(url, str):
                ex.urls["api"][section] = url.replace("https://api.binance.com", f"https://{host}")
            elif isinstance(url, dict):
                ex.urls["api"][section] = {
                    k: (v.replace("https://api.binance.com", f"https://{host}") if isinstance(v, str) else v)
                    for k, v in url.items()
                }
        assert "data-api.binance.vision" in json.dumps(ex.urls["api"]), "mirror remap failed"
    return ex


def fetch_closed(exchange, symbol, timeframe, limit=1000):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.iloc[:-1].reset_index(drop=True)


def signals_for(df, strategy):
    fn, grid = bot.STRATEGIES[strategy]
    sig = fn(df, **grid[0])
    last = sig.iloc[-1]
    return None if pd.isna(last) else bool(last)


# --- virtual wallet -----------------------------------------------------

def default_wallet(balance):
    return {"cash": balance, "qty": 0.0, "entry": 0.0,
            "n_trades": 0, "fees_paid": 0.0, "last_ts": None, "started": None,
            "day": None, "day_start_equity": None, "halted": False}


def slug(symbol):
    return symbol.replace("/", "_")


def state_path(strategy, symbol=None):
    if symbol is None:
            return os.path.join(PAPER_DIR, f"state_{strategy}.json")
    return os.path.join(PAPER_DIR, f"state_{slug(symbol)}_{strategy}.json")
    return os.path.join(PAPER_DIR, f"state_{slug(symbol)}_{strategy}.json")


def read_wallet(strategy, balance, symbol=None):
    p = state_path(strategy, symbol)
    w = default_wallet(balance)
    if os.path.exists(p):
        with open(p) as f:
            w.update(json.load(f))
    return w


def write_wallet(strategy, wallet, symbol=None):
    os.makedirs(PAPER_DIR, exist_ok=True)
    tmp = state_path(strategy, symbol) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(wallet, f, indent=2)
    os.replace(tmp, state_path(strategy, symbol))


def wallet_equity(wallet, price):
    return wallet["cash"] + wallet["qty"] * price


def apply_signal(wallet, price, want_long):
    """All-in long / flat, fee + adverse slippage on both sides."""
    traded, side = False, None
    if want_long and wallet["qty"] == 0.0 and wallet["cash"] > 0:
        fill = price * (1 + bot.SLIPPAGE)
        notional = wallet["cash"]
        wallet["qty"] = notional * (1 - bot.FEE) / fill
        wallet["fees_paid"] += notional * bot.FEE
        wallet["cash"] = 0.0
        wallet["entry"] = fill
        wallet["n_trades"] += 1
        traded, side = True, "BUY"
    elif not want_long and wallet["qty"] > 0.0:
        fill = price * (1 - bot.SLIPPAGE)
        notional = wallet["qty"] * fill
        wallet["cash"] = notional * (1 - bot.FEE)
        wallet["fees_paid"] += notional * bot.FEE
        wallet["qty"] = 0.0
        wallet["entry"] = 0.0
        wallet["n_trades"] += 1
        traded, side = True, "SELL"
    return traded, side


# --- logging ------------------------------------------------------------

def append_csv(path, columns, row):
    os.makedirs(PAPER_DIR, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(columns) + "\n")
        f.write(",".join(str(row[c]) for c in columns) + "\n")


SUMMARY_COLUMNS = ["ts_iso", "ts_ms", "close"]
TRADES_COLUMNS = ["ts_iso", "ts_ms", "symbol", "strategy", "side", "price", "qty", "notional", "fee"]


def summary_path(symbol):
    return os.path.join(PAPER_DIR, f"summary_{slug(symbol)}.csv")


def trades_path(symbol):
    return os.path.join(PAPER_DIR, f"trades_{slug(symbol)}.csv")


def write_summary(cfg, symbol, wallets, actions, close, ts_ms, ts_iso):
    columns = SUMMARY_COLUMNS + [f"eq_{s}" for s in cfg["strategies"]] + \
              [f"in_{s}" for s in cfg["strategies"]]
    row = {"ts_iso": ts_iso, "ts_ms": ts_ms, "close": close}
    for s in cfg["strategies"]:
        row[f"eq_{s}"] = round(wallet_equity(wallets[s], close), 4)
        row[f"in_{s}"] = int(wallets[s]["qty"] > 0)
    append_csv(summary_path(symbol), columns, row)
    if symbol == cfg["symbols"][0]:
        append_csv(SUMMARY_CSV, columns, row)


def write_trade(ts_iso, ts_ms, symbol, strategy, side, price, qty, notional, fee):
    row = {"ts_iso": ts_iso, "ts_ms": ts_ms, "symbol": symbol, "strategy": strategy, "side": side,
        "price": round(price, 6), "qty": round(qty, 8), "notional": round(notional, 4), "fee": round(fee, 6)}
    append_csv(trades_path(symbol), TRADES_COLUMNS, row)



# --- one cycle ----------------------------------------------------------

def step(exchange, cfg, symbol, wallets):
    df = fetch_closed(exchange, symbol, cfg["timeframe"])
    last = df.iloc[-1]
    ts_ms, close = int(last["ts"]), float(last["close"])
    ts_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")

    if all(w["last_ts"] == ts_ms for w in wallets.values()):
        return False

    actions = {}
    for s in cfg["strategies"]:
        w = wallets[s]
        if w["last_ts"] == ts_ms:
            actions[s] = ("SKIP", wallet_equity(w, close))
            continue
        if w["started"] is None:
            w["started"] = ts_ms
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        eq_now = wallet_equity(w, close)
        if w.get("day") != today:
            w["day"] = today
            w["day_start_equity"] = eq_now
            w["halted"] = False
        mdl = float(os.environ.get("MAX_DAILY_LOSS", "0.03"))
        start_eq = w.get("day_start_equity") or eq_now
        if not w["halted"] and eq_now < start_eq * (1 - mdl):
            w["halted"] = True

        want = signals_for(df, s)
        if w["halted"]:
            if w["qty"] > 0:
                apply_signal(w, close, False)
                state_lbl, side_lbl = "HALT", "SELL"
            else:
                state_lbl, side_lbl = "HALT", None
            w["last_ts"] = ts_ms
            write_wallet(s, w, symbol)
            actions[s] = (state_lbl, wallet_equity(w, close))
            continue
        if want is None:
            w["last_ts"] = ts_ms
            write_wallet(s, w, symbol)
            actions[s] = ("WARM", wallet_equity(w, close))
            continue
        traded, side = apply_signal(w, close, want)
        w["last_ts"] = ts_ms
        if traded:
            notional = w["qty"] * close if side == "BUY" else w["cash"]
            write_trade(ts_iso, ts_ms, symbol, s, side, close,
                        w["qty"] if side == "BUY" else 0.0, notional,
                        notional * bot.FEE)
        write_wallet(s, w, symbol)
        state = side if traded else ("LONG" if w["qty"] > 0 else "FLAT")
        actions[s] = (state, wallet_equity(w, close))

    write_summary(cfg, symbol, wallets, actions, close, ts_ms, ts_iso)

    detail = " | ".join(
        f"{s} {a[0]} eq={a[1]:.2f}" for s, a in actions.items())
    log(f"{symbol} {cfg['timeframe']} close={close:.2f} | {detail}")
    return True


# --- runs ---------------------------------------------------------------

def run_once(cfg):
    exchange = build_exchange()
    try:
        for symbol in cfg["symbols"]:
            try:
                wallets = {s: read_wallet(s, cfg["balance"], symbol) for s in cfg["strategies"]}
                step(exchange, cfg, symbol, wallets)
            except Exception as e:
                log(f"{symbol} step failed: {e}")
    finally:
        exchange.close() if hasattr(exchange, "close") else None


def live_tick(exchange, cfg):
    """Lightweight per-minute update: current price + live equity per strategy per symbol."""
    out = {}
    for symbol in cfg["symbols"]:
        try:
            price = float(exchange.fetch_ticker(symbol)["last"])
        except Exception:
            continue
        wallets = {s: read_wallet(s, cfg["balance"], symbol) for s in cfg["strategies"]}
        out[symbol] = {
            "price": round(price, 2),
            "equity": {s: round(wallet_equity(w, price), 2) for s, w in wallets.items()},
        }
    return {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **out} if out else None


def sync_paper(cfg, live=None):
    """Export + push to the live dashboard repo. Lazy import to avoid the
    sync -> multi circular dependency."""
    try:
        import sync
    except ImportError as e:
        log(f"sync import failed (dashboard data not pushed): {e}")
        return
    try:
        sync.setup_sync()
        ok = sync.sync(cfg, commit_msg=f"data: candle {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                       live=live)
        if not ok:
            log("dashboard sync failed (see above)")
    except Exception:
        import traceback
        log("dashboard sync error:\n" + traceback.format_exc())


def run_forever(cfg):
    exchange = build_exchange()
    last_min = None
    while True:
        try:
            any_changed = False
            for symbol in cfg["symbols"]:
                try:
                    wallets = {s: read_wallet(s, cfg["balance"], symbol) for s in cfg["strategies"]}
                    changed = step(exchange, cfg, symbol, wallets)
                    any_changed = any_changed or changed
                except Exception as e:
                    log(f"{symbol} step failed: {e}")
            if any_changed:
                sync_paper(cfg)
        except ccxt.NetworkError as e:
            log(f"NETWORK error, retrying next cycle: {e}")
        except ccxt.ExchangeError as e:
            log(f"EXCHANGE error, retrying next cycle: {e}")
        except Exception:
            import traceback
            log("UNEXPECTED error:\n" + traceback.format_exc())
            raise

        minute = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        if minute != last_min:
            last_min = minute
            try:
                live = live_tick(exchange, cfg)
                if live:
                    sync_paper(cfg, live=live)
            except Exception:
                import traceback
                log("live tick error:\n" + traceback.format_exc())

        time.sleep(30)


# --- comparison report --------------------------------------------------

def max_drawdown_pct(series):
    peak, dd = 0.0, 0.0
    for e in series:
        peak = max(peak, e)
        if peak > 0:
            dd = max(dd, (peak - e) / peak)
    return dd * 100


def report(cfg):
    if not os.path.exists(SUMMARY_CSV):
        print("No data yet. Run `python3 multi.py --once` first.")
        return
    summary = pd.read_csv(SUMMARY_CSV)
    trades = pd.read_csv(TRADES_CSV) if os.path.exists(TRADES_CSV) else None

    start_close = summary["close"].iloc[0]
    end_close = summary["close"].iloc[-1]
    bh = (end_close / start_close - 1) * 100
    period = f"{summary['ts_iso'].iloc[0]}  →  {summary['ts_iso'].iloc[-1]}  ({len(summary)} candles)"

    print("=" * 86)
    print("MULTI-STRATEGY PAPER COMPARISON")
    print(f"{cfg['symbol']} {cfg['timeframe']} · {period}")
    print(f"Buy & Hold benchmark: {bh:+.2f}%")
    print("=" * 86)

    header = (f"{'strategy':<14} {'PnL %':>9} {'vs B&H':>9} {'maxDD %':>8} "
              f"{'trades':>7} {'fee$':>8} {'win %':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for s in cfg["strategies"]:
        col = f"eq_{s}"
        if col not in summary.columns:
            continue
        eq = summary[col].dropna()
        if len(eq) == 0:
            continue
        pnl = (eq.iloc[-1] / cfg["balance"] - 1) * 100
        dd = max_drawdown_pct(eq.tolist())

        st = trades[trades["strategy"] == s] if trades is not None else pd.DataFrame()
        n = 0 if st.empty else len(st)
        fees = 0.0 if st.empty else st["fee"].sum()

        wins = 0
        if not st.empty:
            long = 0.0
            for _, t in st.iterrows():
                if t["side"] == "BUY":
                    long = t["price"]
                else:
                    if long:
                        wins += 1 if t["price"] > long else 0
                        long = 0.0
            win_rate = wins / (n / 2) * 100 if n else 0.0
        else:
            win_rate = 0.0

        rows.append((s, pnl, dd, n, fees, win_rate))

    rows.sort(key=lambda r: -r[1])
    for s, pnl, dd, n, fees, wr in rows:
        print(f"{s:<14} {pnl:>+9.1f} {pnl - bh:>+9.1f} {dd:>8.1f} "
              f"{n:>7} {fees:>8.2f} {wr:>6.0f}%")

    print("-" * len(header))
    print("PnL% is net of 0.1% fee + 0.05% slippage per side. "
          "vs B&H = strategy minus buy&hold over the same window.")


# --- reset --------------------------------------------------------------

def reset():
    if os.path.isdir(PAPER_DIR):
        import shutil
        shutil.rmtree(PAPER_DIR)
        print(f"Removed {PAPER_DIR}/ — fresh start.")
    else:
        print("Nothing to reset.")


# --- checks -------------------------------------------------------------

def demo():
    import tempfile
    global PAPER_DIR, SUMMARY_CSV, TRADES_CSV, PAPER_LOG
    tmp = tempfile.mkdtemp()
    PAPER_DIR, SUMMARY_CSV, TRADES_CSV, PAPER_LOG = \
        tmp, os.path.join(tmp, "summary.csv"), os.path.join(tmp, "trades.csv"), os.path.join(tmp, "paper.log")

    w = default_wallet(1000.0)
    assert w["cash"] == 1000.0 and w["qty"] == 0.0

    traded, side = apply_signal(w, 100.0, True)
    assert traded and side == "BUY" and w["qty"] > 0 and w["cash"] == 0.0
    assert w["fees_paid"] == 1000.0 * bot.FEE

    traded, side = apply_signal(w, 110.0, False)
    assert traded and side == "SELL" and w["qty"] == 0.0 and w["cash"] > 0
    assert w["n_trades"] == 2

    bought = 1000.0 * (1 - bot.FEE) / (100.0 * (1 + bot.SLIPPAGE))
    cash_after = bought * (110.0 * (1 - bot.SLIPPAGE)) * (1 - bot.FEE)
    assert abs(w["cash"] - cash_after) < 1e-6, (w["cash"], cash_after)

    traded, _ = apply_signal(w, 110.0, False)
    assert not traded, "must not re-sell when already flat"

    cfg = {"symbols": ["BTC/USDT"], "symbol": "BTC/USDT", "timeframe": "1h",
           "strategies": ["supertrend", "sma_trend"], "balance": 1000.0}
    wallets = {s: default_wallet(1000.0) for s in cfg["strategies"]}

    class FakeEx:
        def fetch_ohlcv(self, *a, **k):
            closes = [100.0 + i for i in range(400)]
            return [[i * 3600000, c, c + 1, c - 1, c, 100] for i, c in enumerate(closes)]

    ok = step(FakeEx(), cfg, "BTC/USDT", wallets)
    assert ok
    assert os.path.exists(summary_path("BTC/USDT")) or os.path.exists(SUMMARY_CSV)
    assert os.path.exists(trades_path("BTC/USDT")) or os.path.exists(TRADES_CSV) or all(
        wallets[s]["n_trades"] == 0 for s in cfg["strategies"])

    summary = pd.read_csv(summary_path("BTC/USDT") if __import__("os").path.exists(summary_path("BTC/USDT")) else SUMMARY_CSV)
    assert len(summary) == 1
    assert f"eq_supertrend" in summary.columns

    ok = step(FakeEx(), cfg, "BTC/USDT", wallets)
    assert not ok, "already processed this candle must be a no-op"

    import shutil
    shutil.rmtree(tmp)
    print("demo() self-checks passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--report" in sys.argv:
        cfg = build_config()
        report(cfg)
    elif "--reset" in sys.argv:
        reset()
    elif "--once" in sys.argv:
        run_once(build_config())
    else:
        run_forever(build_config())
