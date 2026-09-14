"""Coeur ACP partage par les deux analyses (profils pays / trajectoires
annuelles). Implemente les formules classiques de l'ACP (convention
Husson/Le, cf. base methodologique) : valeurs propres, coordonnees,
correlations variables-composantes (cercle des correlations calcule de
maniere exacte par correlation empirique, pas par simple mise a l'echelle
des vecteurs propres), contributions et cos^2.

Rappel important (cf. consigne) : PCA de scikit-learn centre mais ne
standardise pas automatiquement -> la standardisation (StandardScaler) est
une etape explicite et obligatoire avant l'ACP sur matrice de correlations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    scaler: StandardScaler
    pca: PCA
    variables: list[str]
    index_labels: pd.Index
    scores: pd.DataFrame            # coordonnees des individus (F1..Fp)
    eigenvalues: pd.DataFrame       # composante, valeur propre, variance expliquee %, cumulee %
    var_correlations: pd.DataFrame  # correlations variables x composantes (cercle des correlations)
    var_contributions: pd.DataFrame # contribution des variables aux axes (%)
    var_cos2: pd.DataFrame          # cos^2 des variables
    ind_contributions: pd.DataFrame # contribution des individus aux axes (%)
    ind_cos2: pd.DataFrame          # cos^2 des individus
    n_obs: int = field(default=0)


def run_pca(
    df_wide: pd.DataFrame,
    variables: list[str],
    index_labels: pd.Index,
    n_components: int | None = None,
) -> PCAResult:
    """Execute une ACP standardisee sur df_wide[variables] (aucune valeur
    manquante autorisee en entree : la gestion des cas incomplets doit etre
    faite en amont par l'appelant, et documentee)."""
    X = df_wide[variables].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("run_pca: valeurs manquantes residuelles dans les variables d'entree.")

    n_obs, n_vars = X.shape
    max_components = min(n_obs - 1, n_vars)
    if n_components is None:
        n_components = max_components
    n_components = min(n_components, max_components)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=None)
    scores_arr = pca.fit_transform(X_std)

    comp_labels = [f"F{i+1}" for i in range(n_components)]
    scores = pd.DataFrame(scores_arr, columns=comp_labels, index=index_labels)

    eigenvalues = pd.DataFrame({
        "composante": comp_labels,
        "valeur_propre": pca.explained_variance_,
        "variance_expliquee_pct": pca.explained_variance_ratio_ * 100,
        "variance_cumulee_pct": np.cumsum(pca.explained_variance_ratio_) * 100,
    })

    # Correlations variables-composantes : calcul empirique direct (exact,
    # independant des conventions de normalisation des vecteurs propres).
    corr_rows = []
    for j, var in enumerate(variables):
        row = {"variable": var}
        for k, comp in enumerate(comp_labels):
            row[comp] = float(np.corrcoef(X_std[:, j], scores_arr[:, k])[0, 1])
        corr_rows.append(row)
    var_correlations = pd.DataFrame(corr_rows).set_index("variable")

    # Contributions des variables (%) : u_jk^2 * 100, exact par construction
    # (sum_j u_jk^2 = 1 pour chaque axe k, vecteurs propres normes).
    loadings = pca.components_.T  # shape (n_vars, n_components) = u_jk
    var_contrib = pd.DataFrame(
        (loadings ** 2) * 100, columns=comp_labels, index=variables,
    )
    var_contrib.index.name = "variable"

    # cos^2 des variables = correlation^2 (qualite de representation)
    var_cos2 = var_correlations.pow(2)

    # Contributions des individus (%) : F_ik^2 / (n * lambda_k) * 100
    eig = pca.explained_variance_.reshape(1, -1)
    ind_contrib_arr = (scores_arr ** 2) / (n_obs * eig) * 100
    ind_contrib = pd.DataFrame(ind_contrib_arr, columns=comp_labels, index=index_labels)

    # cos^2 des individus = F_ik^2 / somme_k'(F_ik'^2) sur toutes les
    # composantes calculees
    row_sq_sum = (scores_arr ** 2).sum(axis=1, keepdims=True)
    row_sq_sum[row_sq_sum == 0] = np.nan
    ind_cos2_arr = (scores_arr ** 2) / row_sq_sum
    ind_cos2 = pd.DataFrame(ind_cos2_arr, columns=comp_labels, index=index_labels)

    return PCAResult(
        scaler=scaler,
        pca=pca,
        variables=variables,
        index_labels=index_labels,
        scores=scores,
        eigenvalues=eigenvalues,
        var_correlations=var_correlations,
        var_contributions=var_contrib,
        var_cos2=var_cos2,
        ind_contributions=ind_contrib,
        ind_cos2=ind_cos2,
        n_obs=n_obs,
    )
