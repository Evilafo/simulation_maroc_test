# Étape 2 — Portefeuille prédictif et validation

Statut global : **implémenté, exécuté, évalué**. Aucun modèle n'est **validé** au sens opérationnel (revue métier DEPF/DTFE, déploiement) : seul le statut « évalué » est atteint pour l'ensemble du portefeuille, conformément à la consigne de ne pas confondre meilleur score observé et validation opérationnelle.

## 1. Portefeuille implémenté

| Famille | Modèles | Statut |
|---|---|---|
| Baselines obligatoires (cadrage Ch.6.5.2) | Marche aléatoire, marche aléatoire avec dérive, AR(p) (p choisi par AIC, plafond 3) | **exécuté** pour les 7 cibles × 5 horizons |
| VAR/VECM (Maroc seul) | 4 systèmes candidats économiquement motivés | **rejeté en amont** pour les 4 (voir `ETAPE2_PROTOCOLE_ET_CARACTERISTIQUES.md` §6) — aucun n'est estimé |
| Elastic Net (Maroc seul) | Prévision directe par horizon, caractéristiques et hyperparamètres gelés sur le bloc interne | **exécuté, évalué** pour les 7 cibles × 5 horizons |
| Panel pooled avec effets fixes pays | Elastic Net pooled sur 9 pays, dummies pays (Maroc inclus dans l'apprentissage) | **exécuté, évalué** pour les 7 cibles × 5 horizons |
| Panel pooled leave-one-country-out | Identique, Maroc **totalement exclu** de l'apprentissage et des transformations (pas d'effet fixe estimable pour un pays absent) ; teste la transférabilité pure | **exécuté, évalué** pour les 7 cibles × 5 horizons |
| BVAR, Boosting/MLP, DFM/MIDAS, SVAR | — | **non implémentés à ce stade** (limite de portée assumée, cf. §4) |

Chaque spécification Elastic Net (35 = 7 cibles × 5 horizons) a ses caractéristiques et hyperparamètres gelés consignés dans `outputs/etape2/models/e2_model_registry.json`, et les objets de modèle (coefficients + scaler + liste ordonnée des caractéristiques) sauvegardés dans `outputs/etape2/models/elasticnet_<cible>_h<horizon>_<interne|finale>.pkl` — un fichier par bloc, jamais écrasé par une réestimation ultérieure.

## 2. Méthode d'évaluation (cadrage Ch.6.5.3, appliqué strictement)

- **Cibles en variation** (CROISSANCE_PIB, INFLATION_CPI, TACH) : RMSE/MAE, seuil de gain minimal de 10 % vs la baseline marche aléatoire.
- **Cibles en niveau/ratio** (SOBG, DETTE_PUBLIQUE, BALANCE_COURANTE, FBCF) : RMSE relative (= RMSE / moyenne(|y_true|), définition explicitée) et biais moyen, avec un indicateur de dérive systématique (biais > 0,5×RMSE).
- **Test de Diebold-Mariano** : implémenté avec la correction petit-échantillon de Harvey-Leybourne-Newbold et une variance de Newey-West (retard = horizon-1, pour les erreurs sérialement corrélées à h>1), comparé à une loi de Student. **Calculé seulement si n ≥ 8 prévisions** (seuil de `configs/config_etape2.yaml::evaluation.dm_test_min_forecasts`) ; sinon explicitement rapporté comme « non calculé » plutôt que produit avec une puissance illusoire.
- **Effectif minimal de conclusion** : sous 4 prévisions, le statut est automatiquement « non concluant ». Le bloc final ne compte que 4 prévisions par (cible, horizon, modèle) — la consigne du cadrage est respectée : aucune conclusion de significativité fine n'est tirée sur ce bloc seul.
- **Couverture d'intervalles (68 %/95 %)** : **non calculée dans cette passe** — nécessiterait une distribution des résidus walk-forward par origine, non construite faute de temps ; limite assumée, à combler avant l'étape 3 si des bandes d'incertitude sont requises pour la simulation.
- **Combinaison de modèles** : **non testée** — le cadrage (Ch.6.5.4) exige une amélioration démontrée à la fois de la performance moyenne et de la variance de l'erreur avant de la retenir ; avec 4 observations finales par combinaison, une telle démonstration ne serait pas crédible. Chaque modèle individuel est rapporté séparément.

## 3. Résultats synthétiques (bloc final 2021-2024, meilleur modèle par cible/horizon)

Table complète : `outputs/etape2/tables/e2_13_meilleur_modele_par_cible_horizon.csv` (35 lignes). Table exhaustive de toutes les combinaisons (cible × horizon × modèle × bloc) : `outputs/etape2/tables/e2_12_verdicts.csv` (210 lignes).

- **Cibles en variation** : 14 combinaisons (cible, horizon) sur 15 atteignent le seuil de gain de 10 % vs marche aléatoire sur le bloc final. Seule exception : TACH à h=1 (gain négatif, -13 %). Le gain le plus net est observé sur CROISSANCE_PIB (35 à 67 % selon l'horizon) — à interpréter avec prudence : le bloc final 2021-2024 contient le rebond post-COVID, où une marche aléatoire ancrée sur 2020 (année de récession de -7 %) produit des erreurs mécaniquement très larges. Ce n'est pas une confirmation que le modèle « comprend » le cycle, seulement qu'il bat une baseline naïve pénalisée par un choc exceptionnel.
- **Cibles en niveau/ratio** : 13 combinaisons sur 20 ont un modèle candidat avec une RMSE relative inférieure à la baseline **et** sans dérive systématique détectée. `DETTE_PUBLIQUE` est la cible la plus difficile : aucun modèle ne bat la marche aléatoire de façon convaincante sur 4 des 5 horizons, avec des biais importants aux horizons longs (h=4, h=5) — cohérent avec le caractère très persistant (proche I(1)) et les dynamiques post-COVID inhabituelles de la dette publique marocaine sur cette période.
- **Aucun test de Diebold-Mariano n'est significatif ni même calculable sur le bloc final** (n=4 systématiquement inférieur au seuil de 8) : tous les gains rapportés ci-dessus sont des **observations descriptives**, pas des résultats statistiquement établis. Sur le bloc interne (10 à 26 observations selon l'horizon), le test DM a pu être calculé pour 84 des 105 combinaisons ; aucune significativité forte et systématique n'y est rapportée non plus (à consulter dans `e2_11_evaluation_vs_baseline.csv`, lignes `bloc=interne`).
- **Répartition du meilleur modèle** : Elastic Net (11/35), panel avec effets fixes (13/35), panel leave-one-country-out (11/35) — aucune famille ne domine systématiquement, cohérent avec un jeu de données trop petit pour trancher définitivement entre approches.

## 4. Limites explicitement non levées

- BVAR, Boosting/MLP, DFM/MIDAS, SVAR non implémentés (portée de cette passe).
- Importance par permutation et SHAP non calculés.
- Couverture empirique des intervalles de confiance non produite.
- Combinaison de modèles non testée (effectif insuffisant pour une comparaison crédible).
- Le panel « avec effets fixes pays » suppose implicitement que la relation entre caractéristiques et cible est homogène entre pays hors effet fixe additif — hypothèse forte, non testée formellement (pas de test d'homogénéité des pentes).
- Aucun vintage réel : le backtest reste qualifié de rétrospectif sur données révisées (limite déjà actée en étape 1, reconduite ici).

## 5. Interfaces réutilisables pour l'étape 3

- `outputs/etape2/models/elasticnet_<cible>_h<horizon>_finale.pkl` : modèle Elastic Net réentraîné sur toutes les données disponibles jusqu'à `2024-horizon`, prêt à produire une prévision pour 2024 (ou à projeter un scénario de donneurs en étape 3, sous réserve de fournir les mêmes caractéristiques dans le même ordre — champ `features` du pickle).
- `outputs/etape2/models/e2_model_registry.json` : caractéristiques gelées, hyperparamètres et détail du réglage, par cible/horizon.
- `src/etape2/features.py::build_feature_frame` et `src/etape2/temporal.py` : réutilisables tels quels pour construire de nouvelles origines de prévision (étape 3) sans dupliquer la logique de non-fuite.
