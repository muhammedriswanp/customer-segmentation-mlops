# MLflow Experiment Report

## Experiment: customer-segmentation

## Runs Summary
| Run      | n_clusters | silhouette | inertia  |
|----------|------------|------------|----------|
| kmeans-k2 | 2         | 0.2859     | 34027.15 |
| kmeans-k3 | 3         | 0.1998     | 30243.64 |
| kmeans-k4 | 4         | 0.1561     | 28671.24 |
| kmeans-k5 | 5         | 0.1618     | 27240.97 |

## Best Run
- **Run:** kmeans-k2
- **Silhouette Score:** 0.2859 (higher = better)
- **Inertia:** 34027.15

## Conclusion
k=2 gives best cluster separation.
However project uses k=3 for business interpretability.