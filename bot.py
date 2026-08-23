"""
Paper-trading backtester: signal-based strategies, fee/slippage aware.
No API keys needed — public OHLCV via ccxt, simulated wallet.

  python3 bot.py --demo     self-checks
  python3 bot.py            baseline SMA crossover table (gate 0)
  python3 bot.py --sweep    strategy search, in-sample tuned / out-of-sample judged
"""
import os
import sys
import time
from dataclasses import dataclass

import ccxt
import numpy as np
import pandas as pd

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]
STARTING_BALANCE = 1000.0

FEE = 0.001        # Binance spot taker, per side
SLIPPAGE = 0.0005  # assumed adverse fill
YEARS = 4
CACHE_DIR = "data"
SPLIT = 0.7        # first 70% tunes, last 30% judges

MS_PER_TF = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


# --- data ---------------------------------------------------------------

def fetch_ohlcv(exchange, symbol, timeframe, years=YEARS):
    """Paginated OHLCV with a CSV cache. ccxt caps each request at 1000 candles."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/{symbol.replace('/', '_')}_{timeframe}_{years}y.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)

    step = MS_PER_TF[timeframe]
    since = exchange.milliseconds() - years * 365 * 24 * 3_600_000
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + step
        if len(batch) < 1000 or since > exchange.milliseconds():
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


# --- strategies ---------------------------------------------------------
# Each returns a "want to be long" series: 1 = hold, 0 = flat.
# Every input is shifted or rolling-closed, so no bar sees its own future.

def sma_cross(df, fast=10, slow=30):
    f = df.close.rolling(fast).mean()
    s = df.close.rolling(slow).mean()
    return (f > s).astype(float).where(s.notna())


def sma_cross_trend(df, fast=10, slow=30, trend=200):
    """Crossover, but only while price is above a long-term average."""
    f = df.close.rolling(fast).mean()
    s = df.close.rolling(slow).mean()
    t = df.close.rolling(trend).mean()
    return ((f > s) & (df.close > t)).astype(float).where(t.notna())


def donchian(df, n=20):
    """Breakout: long above the prior n-bar high, flat below the prior n-bar low.

    Fewer, longer trades than a crossover — which is the point, fees scale with
    trade count.
    """
    upper = df.close.rolling(n).max().shift(1)
    lower = df.close.rolling(n).min().shift(1)
    sig = pd.Series(float("nan"), index=df.index)
    sig[df.close > upper] = 1.0
    sig[df.close < lower] = 0.0
    return sig.ffill().where(upper.notna())


def rsi(close, n=14):
    """Wilder's RSI. Causal — each value uses only bars up to and including its own."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, float("nan")))


def rsi_reversion(df, n=14, oversold=30, overbought=70):
    """Buy oversold, sell overbought — the classic retail mean-reversion trade."""
    r = rsi(df.close, n)
    sig = pd.Series(float("nan"), index=df.index)
    sig[r < oversold] = 1.0
    sig[r > overbought] = 0.0
    return sig.ffill().where(r.notna())


def trend_200(df, trend=200):
    """Hold only above the long average. Very few trades, so very little fee drag."""
    t = df.close.rolling(trend).mean()
    return (df.close > t).astype(float).where(t.notna())


def atr(df, n=14):
    high, low, close = df.high, df.low, df.close
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def supertrend(df, atr_period=10, multiplier=3.0):
    """ATR-based trailing stop / trend follower."""
    a = atr(df, atr_period)
    hl2 = (df.high + df.low) / 2
    upper_band = hl2 + (multiplier * a)
    lower_band = hl2 - (multiplier * a)

    in_uptrend = True
    signals = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        if i < atr_period:
            continue
        c = df.close.iloc[i]
        lb = lower_band.iloc[i]
        ub = upper_band.iloc[i]

        if c > (upper_band.iloc[i - 1] if i > 0 else ub):
            in_uptrend = True
        elif c < (lower_band.iloc[i - 1] if i > 0 else lb):
            in_uptrend = False

        signals.iloc[i] = 1.0 if in_uptrend else 0.0

    return signals.where(a.notna())


def bb_squeeze(df, n=20, squeeze_pct=0.08):
    """Bollinger Band squeeze & volatility expansion."""
    mid = df.close.rolling(n).mean()
    std = df.close.rolling(n).std()
    upper = mid + 2.0 * std
    bandwidth = (upper - (mid - 2.0 * std)) / mid
    tight = bandwidth.rolling(10).min() < squeeze_pct

    sig = pd.Series(float("nan"), index=df.index)
    sig[tight & (df.close > upper)] = 1.0
    sig[df.close < mid] = 0.0
    return sig.ffill().fillna(0.0).where(upper.notna())


def macd_strategy(df, fast=12, slow=26, sig=9):
    """MACD histogram momentum: long saat histogram positif."""
    ema_f = df.close.ewm(span=fast, adjust=False).mean()
    ema_s = df.close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal
    out = pd.Series(float("nan"), index=df.index)
    out[hist > 0] = 1.0
    out[hist < 0] = 0.0
    return out


def keltner_break(df, n=20, mult=1.5):
    """Breakout kanal Keltner: entry close menembus band atas, exit di bawah EMA."""
    ema = df.close.ewm(span=n, adjust=False).mean()
    a = atr(df, n)
    upper = ema + mult * a
    sig = pd.Series(float("nan"), index=df.index)
    sig[df.close > upper.shift(1)] = 1.0
    sig[df.close < ema] = 0.0
    return sig.ffill().fillna(0.0).where(upper.notna())


def pctb_revert(df, n=20, k=2.0):
    """Mean reversion volatilitas: beli saat %B < 0 (di bawah band bawah), exit saat kembali > 0.5."""
    mid = df.close.rolling(n).mean()
    sd = df.close.rolling(n).std()
    lower = mid - k * sd
    upper = mid + k * sd
    pb = (df.close - lower) / (upper - lower)
    sig = pd.Series(float("nan"), index=df.index)
    sig[pb < 0] = 1.0
    sig[pb > 0.5] = 0.0
    return sig.ffill().fillna(0.0).where(pb.notna())


def roc_mom(df, n=24, enter=0.02, exit_th=-0.02):
    """Rate-of-change momentum: entry saat return n-bar > ambang, exit saat balik dalam negatif."""
    roc = df.close.pct_change(n)
    sig = pd.Series(float("nan"), index=df.index)
    sig[roc > enter] = 1.0
    sig[roc < exit_th] = 0.0
    return sig.ffill().fillna(0.0)


def obv_slope(df, n=24):
    """/On-Balance-Volume regime: long saat slope OBV n-bar positif (volume mengonfirmasi tren)."""
    direction = np.sign(df.close.diff()).fillna(0.0)
    obv = (direction * df.volume).cumsum()
    slope = obv - obv.shift(n)
    sig = pd.Series(0.0, index=df.index)
    sig[slope > 0] = 1.0
    return sig.where(slope.notna())


STRATEGIES = {
    "supertrend": (supertrend, [{"atr_period": p, "multiplier": m}
                                for p in (10, 14, 20) for m in (2.0, 3.0, 4.0)]),
    "bb_squeeze": (bb_squeeze, [{"n": n, "squeeze_pct": sq}
                                for n in (20, 30) for sq in (0.05, 0.08, 0.12)]),
    "sma_cross": (sma_cross, [{"fast": f, "slow": s}
                              for f in (10, 20, 50) for s in (30, 100, 200) if f < s]),
    "sma_trend": (sma_cross_trend, [{"fast": f, "slow": s, "trend": 200}
                                    for f in (10, 20, 50) for s in (30, 100) if f < s]),
    "donchian": (donchian, [{"n": n} for n in (20, 30, 55, 100)]),
    "rsi_revert": (rsi_reversion, [{"n": 14, "oversold": o, "overbought": 100 - o}
                                   for o in (20, 25, 30, 35)]),
    "trend_200": (trend_200, [{"trend": t} for t in (100, 150, 200)]),
    "macd": (macd_strategy, [{"fast": 12, "slow": 26, "sig": 9}]),
    "keltner": (keltner_break, [{"n": 20, "mult": 1.5}, {"n": 20, "mult": 2.0}]),
    "pctb_revert": (pctb_revert, [{"n": 20, "k": 2.0}, {"n": 14, "k": 2.0}]),
    "roc_mom": (roc_mom, [{"n": 24, "enter": 0.02, "exit_th": -0.02},
                          {"n": 48, "enter": 0.03, "exit_th": -0.03}]),
    "obv_slope": (obv_slope, [{"n": 24}, {"n": 48}]),
}


# --- simulation ---------------------------------------------------------

@dataclass
class Position:
    entry_price: float
    qty: float


def simulate(df, signal):
    """Long-only, acts on signal transitions. Returns (trades, balance, equity).

    Costs are charged both sides: slippage moves the fill against us, fee comes
    off the notional. Strip these and almost any strategy looks profitable.
    """
    balance = STARTING_BALANCE
    position: Position | None = None
    trades, equity = [], []
    prev_sig = 0.0

    for i in range(len(df)):
        sig = signal.iloc[i]
        price = df.close.iloc[i]
        if pd.isna(sig):
            equity.append(balance)
            continue

        if sig == 1.0 and prev_sig == 0.0 and position is None:
            fill = price * (1 + SLIPPAGE)
            qty = balance * (1 - FEE) / fill
            position = Position(fill, qty)
            balance = 0.0
            trades.append({"ts": df.ts.iloc[i], "side": "BUY", "price": fill, "qty": qty})
        elif sig == 0.0 and prev_sig == 1.0 and position is not None:
            fill = price * (1 - SLIPPAGE)
            balance = position.qty * fill * (1 - FEE)
            trades.append({"ts": df.ts.iloc[i], "side": "SELL", "price": fill, "qty": position.qty})
            position = None

        prev_sig = sig
        equity.append(balance if position is None else position.qty * price)

    if position is not None:
        balance = position.qty * df.close.iloc[-1] * (1 - FEE)  # mark-to-market

    return trades, balance, equity


# --- metrics ------------------------------------------------------------

def round_trips(trades):
    pnls = []
    for buy, sell in zip(trades[::2], trades[1::2]):
        if buy["side"] == "BUY" and sell["side"] == "SELL":
            pnls.append(sell["qty"] * sell["price"] * (1 - FEE)
                        - buy["qty"] * buy["price"] / (1 - FEE))
    return pnls


def metrics(trades, final_balance, equity):
    pnls = round_trips(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    peak, max_dd = 0.0, 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)

    gross_win, gross_loss = sum(wins), abs(sum(losses))
    return {
        "trades": len(pnls),
        "pnl_pct": (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100,
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0.0,
        "max_dd": max_dd * 100,
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
    }


def run(df, fn, params):
    return metrics(*simulate(df, fn(df, **params)))


def buy_and_hold(df):
    """The benchmark that matters. Beating cash is easy; beating this is the job."""
    return (df.close.iloc[-1] / df.close.iloc[0] - 1) * 100


# --- runners ------------------------------------------------------------

def load_all(exchange, timeframes=TIMEFRAMES):
    data = {}
    for symbol in PAIRS:
        for tf in timeframes:
            try:
                data[(symbol, tf)] = fetch_ohlcv(exchange, symbol, tf)
            except Exception as e:
                print(f"  fetch failed {symbol} {tf}: {e}")
    return data


def backtest():
    """Gate 0: the original SMA(10/30) idea, now paying real costs."""
    data = load_all(ccxt.binance({"enableRateLimit": True}), ["1h", "4h"])
    print(f"SMA(10/30) long-only · fee {FEE:.2%}/side · slippage {SLIPPAGE:.2%} · {YEARS}y\n")
    header = f"{'pair':<10} {'tf':<4} {'trades':>7} {'PnL %':>9} {'win %':>7} {'maxDD %':>8} {'PF':>6}"
    print(header)
    print("-" * len(header))

    results = []
    for (symbol, tf), df in data.items():
        m = run(df, sma_cross, {"fast": 10, "slow": 30})
        results.append(m)
        print(f"{symbol:<10} {tf:<4} {m['trades']:>7} {m['pnl_pct']:>+9.1f} "
              f"{m['win_rate']:>7.1f} {m['max_dd']:>8.1f} {m['profit_factor']:>6.2f}")

    profitable = sum(1 for m in results if m["pnl_pct"] > 0)
    avg = sum(m["pnl_pct"] for m in results) / len(results)
    print(f"\nGATE 0: {profitable}/{len(results)} profitable · avg PnL {avg:+.1f}% · "
          f"worst DD {max(m['max_dd'] for m in results):.1f}%")
    print("VERDICT:", "PASS" if profitable > len(results) / 2 and avg > 0 else "FAIL")


def sweep():
    """T5: tune each strategy on the first 70% of history, judge on the last 30%.

    Parameters are picked without ever seeing the test slice, so the out-of-sample
    column is the only number worth believing.
    """
    data = load_all(ccxt.binance({"enableRateLimit": True}))
    print(f"Tune on first {SPLIT:.0%} of {YEARS}y, judge on the rest · "
          f"fee {FEE:.2%}/side · slippage {SLIPPAGE:.2%}\n")

    summary = {}
    for name, (fn, grid) in STRATEGIES.items():
        header = (f"{'pair':<10} {'tf':<4} {'best params':<28} "
                  f"{'IS PnL%':>8} {'OOS PnL%':>9} {'B&H%':>8} {'OOS DD%':>8} {'OOS n':>6}")
        print(f"=== {name} ===")
        print(header)
        print("-" * len(header))

        oos_results = []
        for (symbol, tf), df in sorted(data.items()):
            cut = int(len(df) * SPLIT)
            train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
            if len(test) < 150:
                continue

            best = max(grid, key=lambda p: run(train, fn, p)["pnl_pct"])
            is_m = run(train, fn, best)
            oos = run(test, fn, best)
            bh = buy_and_hold(test)
            oos_results.append((oos, bh))

            print(f"{symbol:<10} {tf:<4} {str(best):<28} {is_m['pnl_pct']:>+8.1f} "
                  f"{oos['pnl_pct']:>+9.1f} {bh:>+8.1f} {oos['max_dd']:>8.1f} {oos['trades']:>6}")

        if oos_results:
            avg = sum(o["pnl_pct"] for o, _ in oos_results) / len(oos_results)
            wins = sum(1 for o, _ in oos_results if o["pnl_pct"] > 0)
            beat_bh = sum(1 for o, b in oos_results if o["pnl_pct"] > b)
            worst_dd = max(o["max_dd"] for o, _ in oos_results)
            summary[name] = (avg, wins, len(oos_results), beat_bh, worst_dd)
            print(f"  OOS: avg {avg:+.1f}% · {wins}/{len(oos_results)} profitable · "
                  f"{beat_bh}/{len(oos_results)} beat buy&hold · worst DD {worst_dd:.1f}%\n")

    print("=== GATE 0 verdict (out-of-sample only) ===")
    for name, (avg, wins, total, beat_bh, dd) in sorted(summary.items(), key=lambda x: -x[1][0]):
        verdict = "PASS" if avg > 0 and wins > total / 2 and beat_bh > total / 2 else "FAIL"
        print(f"{name:<12} avg {avg:>+7.1f}%  profitable {wins}/{total}  "
              f"beat B&H {beat_bh}/{total}  worstDD {dd:>5.1f}%  {verdict}")


# --- checks -------------------------------------------------------------

def demo():
    global FEE, SLIPPAGE
    flat = [100.0] * 35
    closes = flat + [100 + i for i in range(35)]
    df = pd.DataFrame({"ts": range(len(closes)), "close": closes})

    trades, bal, _ = simulate(df, sma_cross(df))
    assert len(trades) == 1 and trades[0]["side"] == "BUY", trades
    assert bal > STARTING_BALANCE, "uptrend must beat costs"
    assert trades[0]["price"] > df.close[df.ts == trades[0]["ts"]].iloc[0], "buy fills above close"

    # costs on vs off, same series — costs must always cost something
    with_costs = simulate(df, sma_cross(df))[1]
    FEE, SLIPPAGE = 0.0, 0.0
    try:
        without_costs = simulate(df, sma_cross(df))[1]
    finally:
        FEE, SLIPPAGE = 0.001, 0.0005
    assert with_costs < without_costs, "fee/slippage must reduce the final balance"

    flat_df = pd.DataFrame({"ts": range(80), "close": [100.0] * 80})
    t, b, _ = simulate(flat_df, sma_cross(flat_df))
    assert t == [] and b == STARTING_BALANCE, "flat market must not trade"

    # no lookahead: truncating the future must not change past signals
    long_df = pd.DataFrame({"ts": range(300), "close": [100 + (i % 40) for i in range(300)]})
    for fn, p in ((sma_cross, {}), (donchian, {}), (sma_cross_trend, {}),
                  (rsi_reversion, {}), (trend_200, {})):
        full = fn(long_df, **p)[:200].reset_index(drop=True)
        trunc = fn(long_df.iloc[:200].reset_index(drop=True), **p)
        assert full.equals(trunc), f"{fn.__name__} peeks at future bars"

    m = metrics([], STARTING_BALANCE, [STARTING_BALANCE] * 10)
    assert m["trades"] == 0 and m["max_dd"] == 0.0
    print("demo() self-checks passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--sweep" in sys.argv:
        sweep()
    else:
        backtest()
