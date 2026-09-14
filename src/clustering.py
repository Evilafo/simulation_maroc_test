"""Typologie des pays par classification (Etape 1, UC-S1).

Compare K-means et classification hierarchique ascendante (CAH) a 2 et 3
classes, sur l'espace des composantes retenues d'une ACP (profils moyens ou
trajectoires). Rend explicite l'espace utilise, les effectifs, le score de
silhouette et une mesure de stabilite (repetitions K-means avec graines
differentes ; comparaison des partitions K-means/CAH par indice de Rand
ajuste). N'impose ni le nombre de classes ni l'appartenance du Maroc a un
groupe donne : ce sont des resultats a interpreter, pas des verites
imposees en amont.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def cluster_comparison(
    coords: pd.DataFrame,
    component_cols: list[str],
    k_values: list[int],
    random_state: int = 42,
    n_init: int = 50,
    n_stability_runs: int = 20,
) -> dict:
    """Retourne, pour chaque k demande, les affectations K-means et CAH, la
    silhouette de chacune, et un indice de stabilite K-means (moyenne des
    ARI entre `n_stability_runs` executions avec graines differentes)."""
    X = coords[component_cols].to_numpy(dtype=float)
    n = X.shape[0]
    results = {}

    Z = linkage(X, method="ward")

    for k in k_values:
        if k >= n:
            continue

        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km_labels = km.fit_predict(X)

        hier_labels = fcluster(Z, t=k, criterion="maxclust")

        sil_km = silhouette_score(X, km_labels) if len(set(km_labels)) > 1 else np.nan
        sil_hier = silhouette_score(X, hier_labels) if len(set(hier_labels)) > 1 else np.nan
        ari_km_hier = adjusted_rand_score(km_labels, hier_labels)

        # Stabilite : plusieurs runs K-means avec graines differentes,
        # compares deux a deux par ARI moyen.
        seed_runs = []
        rng = np.random.RandomState(random_state)
        for _ in range(n_stability_runs):
            seed = int(rng.randint(0, 1_000_000))
            labels_run = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
            seed_runs.append(labels_run)
        aris = []
        for i in range(len(seed_runs)):
            for j in range(i + 1, len(seed_runs)):
                aris.append(adjusted_rand_score(seed_runs[i], seed_runs[j]))
        stability_ari_mean = float(np.mean(aris)) if aris else np.nan

        assign = coords[[]].copy()
        assign["kmeans_cluster"] = km_labels
        assign["hierarchical_cluster"] = hier_labels

        results[k] = {
            "assignments": assign,
            "silhouette_kmeans": sil_km,
            "silhouette_hierarchical": sil_hier,
            "ari_kmeans_vs_hierarchical": ari_km_hier,
            "stability_ari_mean_kmeans": stability_ari_mean,
            "effectifs_kmeans": pd.Series(km_labels).value_counts().sort_index().to_dict(),
            "effectifs_hierarchical": pd.Series(hier_labels).value_counts().sort_index().to_dict(),
        }
    return results


def summarize_cluster_comparison(results: dict, espace_label: str) -> pd.DataFrame:
    rows = []
    for k, res in results.items():
        rows.append({
            "espace": espace_label,
            "k": k,
            "silhouette_kmeans": res["silhouette_kmeans"],
            "silhouette_hierarchical": res["silhouette_hierarchical"],
            "ari_kmeans_vs_hierarchical": res["ari_kmeans_vs_hierarchical"],
            "stabilite_ari_moyen_kmeans": res["stability_ari_mean_kmeans"],
            "effectifs_kmeans": res["effectifs_kmeans"],
            "effectifs_hierarchical": res["effectifs_hierarchical"],
        })
    return pd.DataFrame(rows)
