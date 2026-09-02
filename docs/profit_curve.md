# Profit-curve / expected-value threshold (stretch goal #1)

## Business assumptions

Stored in `src/config/config.yaml`:

- `intervention_cost: 5` — cost of acting on a session the model flags as likely to buy (e.g. discount / outreach)
- `conversion_value: 50` — value of correctly catching one purchase

Net expected value at a threshold:

`(true positives × conversion_value) − (false positives × intervention_cost)`

## Result (held-out test split, same seed as training)

With those costs, the profit curve peaks around **threshold = 0.14** (expected value ≈ **16,545**).

Week 2’s operating point (**threshold = 0.5**) still works as a precision/recall-balanced default, but under these costs it scores lower expected value (≈ **14,030**) because it is more conservative about flagging sessions.

## How to view it

- Notebook: `notebooks/05_profit_curve.ipynb`
- Dashboard: **Performance Metrics** page (inputs to tweak costs live)
- Helper: `src/evaluation/profit.py`

Default `threshold` in config stays at **0.5** on purpose — the profit-optimal value depends on the cost assumptions you choose.
