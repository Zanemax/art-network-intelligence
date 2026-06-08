# Backtest Summary

- Target: `institutional_success_3y`
- Prediction years: 2017-2023
- Folds evaluated: 6

## Average Metrics

| model_type | model | roc_auc | precision_at_k | recall_at_k | average_precision |
| --- | --- | --- | --- | --- | --- |
| graph_model | random_forest | 0.8333 | 0.4667 | 1.0000 | 0.7500 |
| baseline | gallery_prestige_baseline | 0.6360 | 0.2333 | 0.6111 | 0.5986 |
| graph_model | gradient_boosting | 0.7278 | 0.4667 | 1.0000 | 0.5944 |
| baseline | simple_weighted_score_baseline | 0.6089 | 0.2333 | 0.6111 | 0.4711 |
| graph_model | logistic_regression | 0.3000 | 0.1333 | 0.2222 | 0.3816 |
| baseline | museum_count_baseline | 0.4604 | 0.2000 | 0.5278 | 0.3333 |
| baseline | random_baseline | 0.4006 | 0.1667 | 0.3611 | 0.2917 |

## Notes

Each fold trains only on prediction dates before the tested year. Labels use the 3-year future window after each prediction date.