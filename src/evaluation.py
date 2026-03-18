from sklearn.metrics import davies_bouldin_score

def get_davies_bouldin(X, labels, model_name="Model"):
    db_score = davies_bouldin_score(X, labels)
    print(f"  {model_name} DB Index: {db_score:.4f} ")
    return db_score

    