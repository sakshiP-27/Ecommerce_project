# API notes

The FastAPI app is in `src/api/main.py`.

When the API is running, the easiest way to try it is the auto docs:

**http://localhost:8000/docs**

You can also hit:

| Method | Path | What it does |
|--------|------|----------------|
| GET | `/health` | Quick alive check → `{"status":"ok"}` |
| POST | `/predict` | Score one session + return SHAP top features |

## Sample `/predict` body

```json
{
  "Administrative": 1,
  "Administrative_Duration": 14.65,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 19,
  "ProductRelated_Duration": 283.88,
  "BounceRates": 0.008,
  "ExitRates": 0.042,
  "PageValues": 68.58,
  "SpecialDay": 0.0,
  "Month": "June",
  "OperatingSystems": 1,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 1,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

## Example response shape

```json
{
  "prediction": "Purchase",
  "probability": 0.716,
  "threshold": 0.5,
  "top_features": [
    {"feature": "numeric__PageValues", "shap_value": 5.3225}
  ]
}
```

## Notes from building this

- Model + pipeline + SHAP explainer load **once at startup** (not per request).
- Engineered features are added inside `/predict` before the pipeline runs.
- Unknown categories (e.g. `Month: "Smarch"`) shouldn’t crash the API because the encoder uses `handle_unknown="ignore"`.
- PowerShell tip: don’t use Linux-style `curl -H ...` — use `Invoke-RestMethod` or just use `/docs`.
