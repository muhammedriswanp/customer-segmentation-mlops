# MLflow Experiment Report

## Experiment: customer-segmentation

---

## Objective
To find the optimal clustering strategy using:
- KMeans
- Agglomerative Clustering

Evaluation Metric:
- Silhouette Score (higher is better)

---

## Experiment 1: KMeans Baseline

### Parameters Tested
- n_clusters: [2, 3, 4, 5]

### Results

| n_clusters | Silhouette Score | Inertia   |
|------------|------------------|----------|
| 2          | 0.2859           | 34027.15 |
| 3          | 0.1998           | 30243.64 |
| 4          | 0.1561           | 28671.24 |
| 5          | 0.1618           | 27240.97 |

### Conclusion
- Best performance at k = 2
- However, k = 3 chosen for business interpretability

---

## Experiment 2: KMeans Hyperparameter Tuning

### Parameters Tested
- n_clusters: [2, 3, 4, 5]
- init: ['k-means++', 'random']
- max_iter: [100, 200, 300]

### Key Observations

#### max_iter
- No impact on results
- Model converged early
- Recommended: max_iter = 100

#### init
- No difference for k=2, k=3
- For k=4 → random performed better

---

## Experiment 3: Agglomerative Clustering

### Parameters Tested
- n_clusters: [2, 3, 4, 5]
- linkage: ['ward', 'complete', 'average']

### Results Summary

| n_clusters | linkage   | Silhouette Score |
|------------|----------|------------------|
| 2          | average  | 0.4857 |
| 3          | average  | 0.3718 |
| 4          | complete | 0.2728 |
| 5          | average  | 0.2155 |

---

## Best Model Overall

- Model: Agglomerative Clustering  
- n_clusters: 2  
- linkage: average  
- Silhouette Score: 0.4857  

---

## Final Decision

| Perspective | Model |
|------------|------|
| Mathematical Best | Agglomerative (k=2, avg) |
| Business Friendly | KMeans (k=3, k-means++) |

---

## Final Recommendation

- Use KMeans (k=3) for interpretability  
- Use Agglomerative (k=2) for performance benchmarking  

---

## Output

- Best model saved at:  
  models/best_model.pkl  

- All runs tracked in MLflow UI

## Day 6 — Flask API

**Endpoint:** POST /predict  
**Model loaded:** KMeans k=3 (models/kmeans_model.pkl)  
**Pipeline:** scaler → PCA → KMeans.predict()

**Sample request:**
{"income": 75000, "recency": 20, "age": 45, ...}

**Sample response:**
{"cluster_id": 1, "segment_name": "High Value Loyalists", "derived": {...}}

**Lessons learned:**
- best_model.pkl and best_kmeans_model.pkl both contained Agglomerative
  because log_experiment.py picks by silhouette score across all models.
  KMeans k=3 was in kmeans_model.pkl — filename did not match content.
- debug=True with watchdog monitors all file changes including site-packages,
  causing server restarts mid-request. Fixed with use_reloader=False.
- scaler.transform() returns numpy array; PCA needs named DataFrame to avoid
  feature name warning. Fixed by wrapping scaled output in pd.DataFrame.