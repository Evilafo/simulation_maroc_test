# Étape 3 — Résultats : scénarios, Monte Carlo et incertitude

Statut global : **implémenté, exécuté, évalué**. Les bandes d'incertitude et scénarios livrés ici sont des références statistiques et des hypothèses configurables — **aucune n'est une prévision institutionnellement validée**.

## 1. Scénarios de trajectoire (Eq. 3.15)

Cinq valeurs d'`α_s` configurées (`configs/config_etape3.yaml::scenarios.alphas`), **fixées ex ante, sans validation institutionnelle présumée** :

| α | Label | Interprétation |
|---|---|---|
| 0,00 | tendance_nationale_seule | Trajectoire du modèle national marocain (étape 2) seule |
| 0,25 | convergence_faible | Léger rapprochement vers le benchmark combiné |
| 0,50 | convergence_moderee | Convergence à parts égales |
| 0,75 | convergence_forte | Rapprochement marqué vers le benchmark |
| 1,00 | benchmark_pur | Trajectoire entièrement déterminée par le benchmark combiné |

**Une hausse d'α n'est pas systématiquement une amélioration** : pour `CROISSANCE_PIB` en 2025, la tendance nationale (5,05 %) est plus dynamique que le benchmark combiné (3,25 %, tiré vers le bas par des économies plus matures comme le Danemark) — un scénario `α` élevé y est donc une hypothèse de **ralentissement volontaire** vers un rythme de croissance de pays de référence plus avancés, pas un gain de performance. À l'inverse pour `DETTE_PUBLIQUE`, un `α` élevé implique une trajectoire de désendettement optimiste, non étayée par la dynamique budgétaire propre du Maroc dans le même scénario (cf. `ETAPE3_BENCHLEARNING_METHODE.md` §8).

Trois scénarios politiques mono-pays (`α=1` avec `ω_b=1` pour un seul donneur) sont produits en regard de la référence statistique — jamais comme prévision centrale (garde-fou cadrage Ch.3.5.1).

## 2. Simulation Monte Carlo jointe (cadrage Ch.6.4)

**Mode complet livré : 10 000 tirages**, graine fixée (42, configurable). Un mode rapide (500 tirages, `--fast`) existe pour l'itération de développement — **ses sorties ne constituent pas la livraison finale** (signalé explicitement dans les journaux et noms de fichiers).

### 2.1 Distribution des chocs et covariance

- Résidus historiques du modèle Elastic Net marocain à horizon 1 (étape 2, blocs interne+finale), pivotés en une matrice annuelle × 7 cibles (18 années disposant d'une observation simultanée sur les 7 cibles).
- Covariance empirique régularisée par retrait (« shrinkage ») vers sa diagonale (intensité 0,3) — **justifié explicitement par le faible effectif** (18 observations pour une matrice 7×7 : la covariance brute serait instable).
- **Simulation strictement jointe** (tirage multivarié, jamais des bandes marginales indépendantes) : un choc de croissance affecte simultanément les 7 cibles selon leur covariance historique, évitant de compter deux fois un même choc macroéconomique commun.
- Deux distributions comparées par sensibilité : gaussienne (paramétrique, covariance ci-dessus) et bootstrap empirique (rééchantillonnage direct des 18 vecteurs de résidus historiques, préservant leur dépendance jointe réelle sans hypothèse de normalité).
- **Horizons d'extrapolation (6-10)** : la covariance de base (horizon 1) est mise à l'échelle proportionnellement à l'horizon (hypothèse de marche aléatoire sur la variance), **hypothèse documentée, pas une estimation directe** — aucune réalisation n'existe à ces horizons pour l'étalonner.

### 2.2 Contrôle de convergence

Quantiles 2,5 %/50 %/97,5 % comparés à 1 000, 5 000, 10 000 et 20 000 tirages (`e3_42_controle_convergence.csv`). Écart relatif maximal observé au quantile 97,5 % entre les différents effectifs de tirage : **≈11 %**, porté principalement par la cible FBCF (la plus bruitée des 7 dans la matrice de résidus). Les autres cibles stabilisent à moins de 5 % d'écart dès 10 000 tirages. **Convergence jugée acceptable mais pas parfaite** — rapportée telle quelle, pas présentée comme une garantie de précision fine.

### 2.3 Bandes d'incertitude

Livrées en unités économiques originales, aux niveaux 68 % et 95 %, pour chaque (scénario α, horizon, distribution, cible) — `outputs/etape3/tables/e3_43_bandes_incertitude.csv`. **La tolérance de ±5 points du cadrage (Ch.6.5.3) sur la couverture empirique n'est pas vérifiable finement ici** : elle suppose un nombre de réalisations suffisant pour estimer un taux de couverture, alors que le bloc final ne compte que 4 observations par horizon (étape 2) — reconduit explicitement comme limite, conformément à la consigne. Les bandes livrées sont donc qualifiées de **« bandes conditionnelles aux hypothèses »** (distribution des chocs, covariance historique, scénario α) plutôt que de couverture statistiquement vérifiée à ce stade.

### 2.4 Incertitudes couvertes et exclues

| Source d'incertitude | Couverte ? |
|---|---|
| Innovations/chocs macroéconomiques (résidus historiques, corrélation jointe) | Oui |
| Incertitude d'estimation des coefficients du modèle de tendance | **Non** — les coefficients Elastic Net gelés sont traités comme fixes, pas rééchantillonnés |
| Incertitude sur les poids statistiques de combinaison | **Non** — ω_b traités comme fixes par scénario, pas ré-estimés à chaque tirage |
| Incertitude de scénario (choix d'α, choix du portefeuille) | **Non couverte par les bandes** — traitée par la présentation de scénarios multiples séparés, pas par une distribution sur α |

Ces exclusions limitent la largeur des bandes rapportées à la seule incertitude d'innovation : les bandes réelles seraient plus larges si l'incertitude des poids et des coefficients était intégrée. Limite explicitement assumée, pas dissimulée.

## 2.5 Artefact visuel attendu : trajectoires en dents de scie

Les figures de trajectoires (`outputs/etape3/figures/e3_scenarios_*.png`) présentent des variations non monotones d'une année projetée à l'autre. C'est une **conséquence directe et attendue** de la méthode de prévision directe par horizon (reconduite de l'étape 2, cf. `docs/ETAPE2_PROTOCOLE_ET_CARACTERISTIQUES.md`) : chaque horizon (1 à 10) est produit par une spécification indépendamment réglée, pas par un enchaînement itératif d'un même modèle pas-à-pas. Ce choix évite l'accumulation d'erreur propre aux méthodes itérées, mais produit des trajectoires visuellement moins lisses. **Ce n'est pas une anomalie de calcul** — signalé ici pour éviter une mauvaise lecture des figures.

## 3. Lecture des trajectoires et limites de transférabilité

Les projections 2025-2034 combinent :
- des horizons **effectivement testés en backtest** (1 à 5, réalisations disponibles jusqu'en 2024) ;
- des horizons d'**extrapolation longue non testés** (6 à 10, 2030-2034) — signalés dans toutes les tables (`horizon_teste_backtest`) et à traiter avec une prudence accrue, en particulier pour les cibles où le backtest (§ méthode, section 6) montre déjà une combinaison de benchlearning moins performante qu'une simple persistance (SOBG, DETTE_PUBLIQUE, BALANCE_COURANTE, TACH).

**Aucune promesse causale** : la référence de benchlearning décrit des trajectoires historiques comparables et une combinaison statistique optimisée pour reproduire le passé marocain — elle n'établit pas qu'adopter les politiques d'un pays donneur produirait les mêmes effets au Maroc (cf. cadrage Ch.6.3 sur les limites causales de l'apprentissage automatique, déjà actées en étape 2).
