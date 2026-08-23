"""
Live runner — Binance spot testnet by default, real exchange only if forced.

  python3 live.py --demo        self-checks, no network, no keys
  python3 live.py --gen-keys    generate Ed25519 keypair for Binance Testnet
  python3 live.py --gen-keys rsa generate RSA keypair for Binance Testnet
  python3 live.py --once        one decision cycle then exit (use this first)
  python3 live.py               run forever

Auth methods supported:
  1. Ed25519 / RSA Key: set API_KEY & PRIVATE_KEY_PATH (or API_SECRET_FILE) in .env
  2. HMAC Key: set API_KEY & API_SECRET in .env

Env (put in .env, which .gitignore already excludes):
  API_KEY               API Key from Binance Testnet
  PRIVATE_KEY_PATH      path to private key file (e.g. test-prv-key.pem)
  API_SECRET            HMAC secret or raw PEM private key
  SYMBOL=BTC/USDT       market to trade
  TIMEFRAME=4h          decision candle
  STRATEGY=sma_trend    key from bot.STRATEGIES
  RISK=0.5              fraction of quote balance to deploy per entry
  MAX_DAILY_LOSS=0.03   kill switch, fraction of the day's starting equity
  LIVE=0                1 = real money. Anything else = testnet.
  DRY=1                 1 = print signals & orders without sending
"""
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import ccxt
import pandas as pd

import bot

STATE_FILE = "state.json"
LOG_FILE = "live.log"


def load_env(path=".env"):
    """Minimal .env reader — avoids a dependency for six lines of parsing."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def resolve_credentials():
    """Load API key and secret/private key (supports HMAC, RSA, and Ed25519 PEM)."""
    api_key = os.environ.get("API_KEY")
    prv_path = os.environ.get("PRIVATE_KEY_PATH") or os.environ.get("API_SECRET_FILE")

    secret = None
    if prv_path and os.path.exists(prv_path):
        with open(prv_path, "r") as f:
            secret = f.read().strip()
    elif os.environ.get("API_SECRET"):
        raw = os.environ.get("API_SECRET", "").strip()
        # If API_SECRET points to a file, read it
        if (raw.endswith(".pem") or raw.endswith(".key")) and os.path.exists(raw):
            with open(raw, "r") as f:
                secret = f.read().strip()
        else:
            # handle escaped newlines if passed in one line
            secret = raw.replace("\\n", "\n")

    return api_key, secret


def generate_keys(algo="ed25519", prv_path="test-prv-key.pem", pub_path="test-pub-key.pem"):
    """Generate Ed25519 or RSA asymmetric keypair for Binance API."""
    algo = algo.lower()
    if algo not in ("ed25519", "rsa"):
        sys.exit(f"Unsupported algorithm '{algo}'. Use 'ed25519' or 'rsa'.")

    print(f"Generating {algo.upper()} keypair...")
    if algo == "ed25519":
        subprocess.run(["openssl", "genpkey", "-algorithm", "ed25519", "-out", prv_path], check=True)
        subprocess.run(["openssl", "pkey", "-pubout", "-in", prv_path, "-out", pub_path], check=True)
    else:  # rsa 2048
        subprocess.run(["openssl", "genrsa", "-out", prv_path, "2048"], check=True)
        subprocess.run(["openssl", "rsa", "-in", prv_path, "-pubout", "-outform", "PEM", "-out", pub_path], check=True)

    # restrict private key permissions
    os.chmod(prv_path, 0o600)

    with open(pub_path, "r") as f:
        pub_content = f.read().strip()

    print(f"\n[+] Private key saved to: {prv_path} (DO NOT SHARE)")
    print(f"[+] Public key saved to:  {pub_path}\n")
    print("=== Copy Public Key Below and Register on Binance Testnet ===")
    print(pub_content)
    print("=============================================================\n")
    print("Next steps:")
    print("1. Go to https://testnet.binance.vision")
    print("2. Register your public key. Binance will give you an API Key.")
    print("3. Add to your .env file:")
    print(f"   API_KEY=<your_binance_api_key>")
    print(f"   PRIVATE_KEY_PATH={prv_path}")
    print("4. Run `python3 live.py --once` to test.\n")


def log(msg):
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# --- state --------------------------------------------------------------

def read_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"in_position": False, "qty": 0.0, "entry": 0.0,
            "day": None, "day_start_equity": None, "halted": False}


def write_state(state):
    """Write via a temp file so a crash mid-write can't corrupt the state."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def reconcile(exchange, state, symbol):
    """Trust the exchange, not the file. A stale file causes double entries."""
    base = symbol.split("/")[0]
    balance = exchange.fetch_balance()
    on_exchange = balance.get(base, {}).get("free", 0.0) or 0.0
    market = exchange.market(symbol)
    dust = market["limits"]["amount"]["min"] or 0.0

    actually_holding = on_exchange > dust
    if actually_holding != state["in_position"]:
        log(f"RECONCILE: state said in_position={state['in_position']}, "
            f"exchange holds {on_exchange} {base} — trusting exchange")
        state["in_position"] = actually_holding
        state["qty"] = on_exchange if actually_holding else 0.0
    return state


def equity(exchange, symbol):
    base, quote = symbol.split("/")
    bal = exchange.fetch_balance()
    price = exchange.fetch_ticker(symbol)["last"]
    return (bal.get(quote, {}).get("free", 0.0) or 0.0) + \
           (bal.get(base, {}).get("free", 0.0) or 0.0) * price


# --- decision -----------------------------------------------------------

def closed_candles(exchange, symbol, timeframe, limit=1000):
    """Drop the final candle — it is still forming and its close will change.

    Acting on an unclosed candle is the classic way a backtest that looked
    profitable turns into a live bot that is not.
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.iloc[:-1].reset_index(drop=True)


def want_long(df, strategy):
    fn, grid = bot.STRATEGIES[strategy]
    signal = fn(df, **grid[0])
    last = signal.iloc[-1]
    return None if pd.isna(last) else bool(last)


# --- orders -------------------------------------------------------------

def enter(exchange, symbol, state, risk, dry):
    quote = symbol.split("/")[1]
    free = exchange.fetch_balance().get(quote, {}).get("free", 0.0) or 0.0
    price = exchange.fetch_ticker(symbol)["last"]
    market = exchange.market(symbol)

    spend = free * risk
    min_cost = (market["limits"]["cost"] or {}).get("min") or 0.0
    if spend < min_cost:
        log(f"SKIP entry: {spend:.2f} {quote} below exchange minimum {min_cost}")
        return state

    amount = float(exchange.amount_to_precision(symbol, spend / price))
    if amount <= 0:
        log("SKIP entry: amount rounds to zero")
        return state

    if dry:
        log(f"DRY buy {amount} {symbol} ~{price}")
        return state

    order = exchange.create_market_buy_order(symbol, amount)
    filled = order.get("filled") or amount
    log(f"BUY  {filled} {symbol} @ ~{order.get('average') or price}")
    state.update(in_position=True, qty=filled, entry=order.get("average") or price)
    return state


def exit_position(exchange, symbol, state, dry):
    base = symbol.split("/")[0]
    free = exchange.fetch_balance().get(base, {}).get("free", 0.0) or 0.0
    amount = float(exchange.amount_to_precision(symbol, free))
    if amount <= 0:
        log("SKIP exit: nothing to sell")
        state.update(in_position=False, qty=0.0)
        return state

    if dry:
        log(f"DRY sell {amount} {symbol}")
        return state

    order = exchange.create_market_sell_order(symbol, amount)
    log(f"SELL {order.get('filled') or amount} {symbol} @ ~{order.get('average')}")
    state.update(in_position=False, qty=0.0, entry=0.0)
    return state


# --- kill switch --------------------------------------------------------

def check_kill_switch(state, current_equity, max_daily_loss):
    """Halt for the rest of the UTC day once the loss limit is hit."""
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("day") != today:
        state.update(day=today, day_start_equity=current_equity, halted=False)
        log(f"New day {today}, starting equity {current_equity:.2f}")
        return state

    start = state.get("day_start_equity") or current_equity
    if start > 0:
        loss = (start - current_equity) / start
        if loss >= max_daily_loss and not state["halted"]:
            state["halted"] = True
            log(f"KILL SWITCH: down {loss:.2%} today (limit {max_daily_loss:.2%}) — halted")
    return state


# --- cycle --------------------------------------------------------------

def cycle(exchange, cfg, state):
    state = reconcile(exchange, state, cfg["symbol"])
    eq = equity(exchange, cfg["symbol"])
    state = check_kill_switch(state, eq, cfg["max_daily_loss"])

    df = closed_candles(exchange, cfg["symbol"], cfg["timeframe"])
    signal = want_long(df, cfg["strategy"])
    log(f"equity={eq:.2f} in_position={state['in_position']} signal={signal} "
        f"last_close={df.close.iloc[-1]}")

    if state["halted"]:
        if state["in_position"]:
            log("Halted — closing position")
            state = exit_position(exchange, cfg["symbol"], state, cfg["dry"])
        write_state(state)
        return state

    if signal is None:
        log("Not enough history for a signal yet")
    elif signal and not state["in_position"]:
        state = enter(exchange, cfg["symbol"], state, cfg["risk"], cfg["dry"])
    elif not signal and state["in_position"]:
        state = exit_position(exchange, cfg["symbol"], state, cfg["dry"])

    write_state(state)
    return state


def build_config():
    load_env()
    live = os.environ.get("LIVE") == "1"
    cfg = {
        "symbol": os.environ.get("SYMBOL", "BTC/USDT"),
        "timeframe": os.environ.get("TIMEFRAME", "4h"),
        "strategy": os.environ.get("STRATEGY", "sma_trend"),
        "risk": float(os.environ.get("RISK", 0.5)),
        "max_daily_loss": float(os.environ.get("MAX_DAILY_LOSS", 0.03)),
        "dry": os.environ.get("DRY") == "1",
        "live": live,
    }
    if cfg["strategy"] not in bot.STRATEGIES:
        sys.exit(f"Unknown STRATEGY={cfg['strategy']}, pick from {list(bot.STRATEGIES)}")
    return cfg


def build_exchange(cfg):
    key, secret = resolve_credentials()
    if not key or not secret:
        sys.exit(
            "Missing credentials! Please set API_KEY and PRIVATE_KEY_PATH (or API_SECRET) in .env.\n"
            "Tip: Run `python3 live.py --gen-keys` to generate an Ed25519 keypair for Binance Testnet."
        )

    exchange = ccxt.binance({"apiKey": key, "secret": secret, "enableRateLimit": True})
    if cfg["live"]:
        log("!!! LIVE MODE — real money !!!")
    else:
        exchange.set_sandbox_mode(True)
        log("Testnet mode (set LIVE=1 for real money)")
    exchange.load_markets()
    return exchange


def main():
    if "--gen-keys" in sys.argv:
        idx = sys.argv.index("--gen-keys")
        algo = sys.argv[idx + 1] if len(sys.argv) > idx + 1 and not sys.argv[idx + 1].startswith("-") else "ed25519"
        generate_keys(algo=algo)
        return

    cfg = build_config()
    exchange = build_exchange(cfg)
    log(f"config: {cfg}")
    state = read_state()
    once = "--once" in sys.argv
    period = bot.MS_PER_TF[cfg["timeframe"]] / 1000

    while True:
        try:
            state = cycle(exchange, cfg, state)
        except ccxt.NetworkError as e:
            log(f"NETWORK error, retrying next cycle: {e}")
        except ccxt.ExchangeError as e:
            log(f"EXCHANGE error, retrying next cycle: {e}")
        except Exception:
            log("UNEXPECTED error:\n" + traceback.format_exc())
            raise
        if once:
            return
        # wake shortly after the next candle closes
        now = time.time()
        time.sleep(period - (now % period) + 5)


# --- checks -------------------------------------------------------------

def demo():
    s = {"day": None, "day_start_equity": None, "halted": False, "in_position": False}
    s = check_kill_switch(s, 1000.0, 0.03)
    assert s["day_start_equity"] == 1000.0 and not s["halted"]

    s = check_kill_switch(s, 980.0, 0.03)
    assert not s["halted"], "2% loss must not halt a 3% limit"

    s = check_kill_switch(s, 960.0, 0.03)
    assert s["halted"], "4% loss must halt"

    # a new day must reset the halt
    s["day"] = "1999-01-01"
    s = check_kill_switch(s, 960.0, 0.03)
    assert not s["halted"] and s["day_start_equity"] == 960.0

    # closed_candles must drop the forming bar
    class FakeEx:
        def fetch_ohlcv(self, *a, **k):
            return [[i, 1, 1, 1, i, 1] for i in range(10)]
    df = closed_candles(FakeEx(), "BTC/USDT", "4h")
    assert len(df) == 9 and df.close.iloc[-1] == 8, "must drop the unclosed candle"

    # state round trip survives a crash-safe write
    global STATE_FILE
    STATE_FILE = "state.test.json"
    try:
        write_state({"in_position": True, "qty": 1.5})
        assert read_state()["qty"] == 1.5
    finally:
        os.path.exists(STATE_FILE) and os.remove(STATE_FILE)
        STATE_FILE = "state.json"

    # resolve_credentials handles file paths and direct secrets
    os.environ["API_KEY"] = "test_key"
    os.environ["API_SECRET"] = "test_secret"
    k, s = resolve_credentials()
    assert k == "test_key" and s == "test_secret"

    print("demo() self-checks passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
