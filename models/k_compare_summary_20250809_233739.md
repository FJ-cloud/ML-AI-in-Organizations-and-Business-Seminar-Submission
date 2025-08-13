# K comparison (test set)

- Split: 2022-08-31
- Order: 0,0,0

## Primary metric: Average Precision (AP) — higher is better

| Model | K | AP | ROC AUC | Brier |
|---|---:|---:|---:|---:|
| Top-30 | 30 | 0.625 | 0.656 | 0.2173 |
| All | 0 | 0.589 | 0.606 | 0.2174 |
| Top-10 | 10 | 0.585 | 0.633 | 0.2116 |

## All features — Precision/Recall @ top percentiles
| % | k | Precision | Recall |
|---:|---:|---:|---:|
| 5 | 107 | 1.000 | 0.174 |
| 10 | 214 | 1.000 | 0.348 |
| 20 | 429 | 0.601 | 0.420 |

## Top-10 features — Precision/Recall @ top percentiles
| % | k | Precision | Recall |
|---:|---:|---:|---:|
| 5 | 107 | 1.000 | 0.174 |
| 10 | 214 | 1.000 | 0.348 |
| 20 | 429 | 0.548 | 0.382 |

## Top-30 features — Precision/Recall @ top percentiles
| % | k | Precision | Recall |
|---:|---:|---:|---:|
| 5 | 107 | 1.000 | 0.174 |
| 10 | 214 | 1.000 | 0.348 |
| 20 | 429 | 0.643 | 0.449 |