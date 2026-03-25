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

## Experiment 2 - Parameter Tuning

### Parameters Tested
- n_clusters: [2, 3, 4, 5]
- init: ['k-means++', 'random']
- max_iter: [100, 200, 300]
- Total runs: 24

### Key Findings

**max_iter:**
- Did not affect results
- Model converged before 100 iterations
- Use iter=100 to save time

**init:**
- For k=2 and k=3 → no difference
- For k=4 → random (0.2058) beat k-means++ (0.1561)

### Results Summary
| Run | silhouette |
|-----|------------|
| k=2, any init, iter=100 | 0.2859 |
| k=4, random, iter=100   | 0.2058 |
| k=3, k-means++, iter=100| 0.1998 |

### Final Recommendation
- Mathematically → k=2, any init, iter=100
- Business wise  → k=3, k-means++, iter=100
