# Forex Quantitative Trading & Risk Engine (DareFX Aegis-Alpha)

## Engine Overview
DareFX Aegis-Alpha (v2.1) is an autonomous institutional trading bot operating primarily on EUR/USD currency pairs. The engine employs machine learning predictions alongside strict quantitative risk guardrails.

## Machine Learning Policy
- **Primary Model:** LightGBM binary classifier trained on historical tick-derived daily bars from 2003 through 2026.
- **Confidence Threshold:** High-conviction threshold set to `0.52` (52% minimum probability). Signals below this threshold are rejected.
- **Trend Confluence:** Institutional alignment via 50-period Exponential Moving Average (EMA-50). BUY orders require price above EMA-50; SELL orders require price below EMA-50.
- **Regime Filter:** ADX minimum threshold set to `32.0`. Sideways and low-volatility chop markets are aggressively filtered out.

## Protective Circuit Breakers
- **Daily Drawdown Cap:** Hard cap at `5.0%`. If daily equity loss reaches 5%, all trading activity is locked until the next daily session at 00:00 UTC.
- **Maximum Peak Drawdown:** Hard shutdown threshold at `15.0%`. Triggers immediate position liquidation and telegram alarms.
- **Accelerated Break-Even (BEP):** Once an open position reaches `+10.0 pips` profit, the Stop Loss is automatically moved to entry price plus a `+1.5 pips` buffer, guaranteeing a risk-free trade.
