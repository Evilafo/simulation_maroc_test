# Étape 3 — Méthode : benchlearning, combinaison et simulation

Statut global : **implémenté, exécuté, évalué**. Aucune validation institutionnelle (comité de gouvernance du cadrage, Ch.3.3.4) à ce stade — les scénarios et pondérations produits ici sont des références statistiques et des hypothèses configurables, pas des décisions actées.

## 1. Équations implémentées (cadrage Ch.3.5, Eq. 3.14-3.15)

Vérifiées page à page (page imprimée 20 / page 30 du PDF `rapport_cadrage_UC-S1_final.pdf`) :

- **Niveau 1 — benchmark combiné** : `X_bench-comb(t+h) = Σ_b ω_b·X_b(t+h)`, avec `ω_b ≥ 0` et `Σ_b ω_b = 1`.
- **Niveau 2 — trajectoire scénarisée** : `X_s(t+h) = (1−α_s)·X_tend_MAR(t+h) + α_s·X_bench-comb(t+h)`, `0 ≤ α_s ≤ 1`.
- **3ᵉ niveau (lissage/inertie)** : explicitement mentionné par le cadrage comme hors périmètre, relevant de phases ultérieures. **Non implémenté** (`dynamic_adjustment_enabled: false`, vérifié au démarrage — toute activation lève une erreur explicite tant qu'il n'est pas formalisé séparément).

Implémentation : `src/etape3/combination.py`. Aucune substitution par une autre dynamique (pas de moyenne géométrique, pas de pondération temporelle) : les deux équations sont appliquées littéralement, en unités économiques originales (% du PIB, % annuel selon la cible).

## 2. Objets, unités, origine

| Objet | Définition | Origine | Unité |
|---|---|---|---|
| `X_tend_MAR(t+h)` | Trajectoire tendancielle marocaine : modèle Elastic Net retenu et gelé en étape 2 (h=1-5) ou même architecture nouvellement réglée (h=6-10, extrapolation) | 2024 | Unité native de la cible (% annuel ou % du PIB) |
| `X_b(t+h)` | Trajectoire tendancielle du pays donneur b, **projetée par son propre modèle** (même architecture Elastic Net directe par horizon) — jamais une réalisation future observée | 2024 | Idem |
| `X_bench-comb(t+h)` | Moyenne pondérée des `X_b(t+h)` | Dérivée | Idem |
| `X_s(t+h)` | Trajectoire du scénario s | Dérivée | Idem |

**Disponibilité temporelle.** Historique 1991-2024 (36 pays × années au périmètre inchangé depuis l'étape 1). Projections 2025-2034 (h=1 à 10). Horizons 1-5 : **testés en backtest** (réalisations disponibles jusqu'en 2024). Horizons 6-10 : **extrapolation non testée**, signalée dans toutes les tables (`horizon_teste_backtest=False`) et dans les figures.

**Exogènes futures.** Aucune variable exogène future n'est utilisée telle quelle dans les modèles de tendance : toutes les caractéristiques sont construites à partir de retards/transformations de séries elles-mêmes projetées ou observées à l'origine (cf. étape 2). Aucun scénario exogène séparé n'a été jugé nécessaire à ce stade — limite à réexaminer si un moteur DFM/MIDAS est introduit ultérieurement.

## 3. Donneurs : éligibilité, portefeuille, distinction des scores

### 3.1 Score S_POC (exploratoire) — jamais confondu avec le score complet S_c

Le score à 7 critères du cadrage (Ch.3.2.2-3.2.3, Eq. 3.3-3.10) nécessite S1 (PIB/habitant PPA), S4 (contraintes en ressources) et S5 (qualité institutionnelle WGI) : **absents des 30 indicateurs de ce projet** (déjà documenté en étape 1). Le score S_POC ne retient que les composantes calculables :

| Composante | Proxy utilisé | Limite documentée |
|---|---|---|
| S2 (structure productive) | Distance euclidienne des parts VAAG/VAIN/VASE (converties en fractions) à celles du Maroc, Eq. 3.4 | Fidèle à la formule du cadrage |
| S3 (ouverture commerciale) | Écart d'OUVERTURE_COM au Maroc, normalisé par l'écart-type inter-pays (τ_O), Eq. 3.5 | τ_O fixé par convention (écart-type observé), pas par une valeur institutionnelle |
| S6 (profondeur des données) | Taux de complétude sur les 30 indicateurs, Eq. 3.9 | Fidèle à la formule |
| S7 (transformation, partiel) | Proxy à une seule dimension : évolution de la part de FBCF entre début et fin de fenêtre | Le cadrage prévoit 5 dimensions (industrialisation, capital humain, diversification, emploi, inclusion) ; 4 non calculables ici |

Poids renormalisés à parts égales (0,25 chacune) sur ces 4 composantes — choix documenté, le cadrage ne fournissant pas de poids pour un score partiel. **Le seuil de présélection S≥0,55 (Eq. 3.11) n'est jamais appliqué à S_POC** (vérifié au démarrage par `config_check.py`) : il s'applique au score complet à 7 critères, non reconstituable ici.

Résultats (`outputs/etape3/tables/e3_01_score_poc.csv`) : les 8 candidats s'échelonnent de 0,47 (Malaisie, pénalisée par une ouverture commerciale extrême et une progression de FBCF nulle sur la fenêtre) à 0,90 (Chili). Le Viêt Nam, malgré son rôle central dans le portefeuille initial du cadrage, obtient un S_POC modéré (0,69), tiré vers le bas par une ouverture commerciale si extrême qu'elle score 0 en similitude avec le Maroc — **ceci n'invalide pas son intérêt narratif** (le cadrage le retient pour d'autres raisons qualitatives, Ch.3.3.1), mais illustre que S_POC est un indicateur partiel, pas un verdict.

### 3.2 Portefeuille : initial vs alternatif optimisé (Eq. 3.12-3.13)

Portefeuille initial du cadrage (VNM, CRI, DNK) comparé à toutes les combinaisons de taille 3 parmi les 8 candidats, selon `B* = argmax[mean(S_POC) − λ·R(B)]` (λ=0,5). La redondance R(B) est approximée par la corrélation des profils moyens **standardisés entre pays** (correction méthodologique documentée : sans standardisation, la corrélation est artificiellement gonflée par une forme de profil commune à la plupart des économies, indépendamment de leur proximité réelle).

Résultat : le portefeuille initial (VNM, CRI, DNK) obtient un objectif de 0,60 ; le meilleur portefeuille alternatif de même taille selon Eq. 3.12 est (CHL, PRT, VNM), objectif 0,65 — **une amélioration marginale**, pas une remise en cause du choix narratif du cadrage. Table complète : `e3_03_portefeuilles_compares.csv` (toutes les combinaisons C(8,3)=56, exhaustif). Portefeuille alternatif à 8 pays également conservé comme référence de sensibilité maximale.

## 4. Poids de combinaison : statistiques vs politiques (garde-fou Ch.3.5.1)

### 4.1 Poids statistiques

Estimés par un programme quadratique sous contraintes (`ω≥0`, `Σω=1`), régularisé (pénalité L2 légère, `l2_regularization=0.01` — justifiée par le faible nombre de donneurs et l'historique court), au sens du contrôle synthétique cité par le cadrage (Abadie et al.) : ils minimisent l'écart quadratique entre la trajectoire marocaine standardisée et une combinaison convexe des trajectoires donneuses standardisées, sur les 7 cibles poolées (objectif commun équilibrant les unités, comme demandé). Implémentation : `src/etape3/weights.py::estimate_common_weights`.

Résultats pour le portefeuille initial (VNM, CRI, DNK) : ω≈(0,16 ; 0,41 ; 0,43) — le Danemark et le Costa Rica dominent la combinaison statistique, le Viêt Nam pèse moins malgré son rôle narratif central, cohérent avec son S_POC plus modéré et sa trajectoire moins corrélée à celle du Maroc sur la période.

### 4.2 Variante par cible et vérification de cohérence

Poids ré-estimés séparément pour chacune des 7 cibles (`e3_05_poids_statistiques_par_cible.csv`). Cohérence vérifiée par corrélation entre les vecteurs de poids de chaque paire de cibles (`e3_06_coherence_poids_par_cible.csv`) : corrélation moyenne proche de zéro (-0,03), signe que **les poids optimaux diffèrent sensiblement d'une cible à l'autre** — la référence "portefeuille commun" n'est pas un résumé fidèle de ce que ferait un ajustement cible par cible. Limite documentée, pas corrigée silencieusement.

### 4.3 Poids politiques

Trois scénarios mono-pays illustratifs (`ω_VNM=1`, `ω_CRI=1`, `ω_DNK=1`), validés au démarrage (somme=1, poids≥0). **Jamais utilisés pour la prévision centrale** — présentés uniquement en regard de la référence statistique, conformément au garde-fou explicite du cadrage.

## 5. Trajectoires tendancielles (Maroc et donneurs)

Réutilisent l'architecture Elastic Net directe par horizon de l'étape 2 (`src/etape2/features.py`, `selection.py`, `temporal.py`, appelés depuis `src/etape3/trend_projection.py`) :
- **Maroc, h=1-5** : spécification gelée et validée en étape 2, rechargée depuis `outputs/etape2/models/e2_model_registry.json`.
- **Maroc, h=6-10** : même architecture, nouveau réglage (jamais évalué en biais/RMSE, aucune réalisation disponible).
- **8 donneurs, h=1-10** : nouveau réglage pour chacun (l'étape 2 ne portait que sur le Maroc). 630 projections produites (9 pays × 7 cibles × 10 horizons), 35 réutilisent une spécification étape 2 gelée, 595 sont nouvellement réglées à l'étape 3.

## 6. Backtest de la référence de benchlearning (h=1-5, protocole étape 2 reconduit)

Principe respecté explicitement : **les valeurs futures réalisées des donneurs ne sont jamais utilisées** pour construire `X_bench-comb(O+h)` pendant le backtest — seules leurs projections tendancielles (mêmes garanties de non-fuite que le modèle national) entrent dans la combinaison. Les poids statistiques sont ré-estimés à **chaque origine** du backtest, sur l'historique accessible uniquement (`e3_22_historique_poids_reestimes.csv`). Les modèles de tendance des donneurs suivent la même convention de gel que le modèle national de l'étape 2 (caractéristiques/hyperparamètres figés une fois sur le bloc interne, coefficients réajustés par origine) — choix documenté de cohérence et de tractabilité, pas une réduction de rigueur (cf. `configs/config_etape3.yaml::temporal_protocol.reestimation_note`).

**Résultat central, honnête** (`outputs/etape3/tables/e3_21_evaluation_benchlearning.csv`, bloc final 2021-2024) : la référence statistique de benchlearning **bat la marche aléatoire uniquement sur CROISSANCE_PIB (27 à 61 % de gain selon l'horizon) et INFLATION_CPI (5 à 38 %)** — deux cibles cycliques à forte composante de co-mouvement international. Sur les cibles structurelles/institutionnelles (SOBG, DETTE_PUBLIQUE, BALANCE_COURANTE, TACH), **la combinaison des donneurs fait sensiblement moins bien qu'une simple persistance** (RMSE relative 2 à 3 fois supérieure à la baseline pour BALANCE_COURANTE ; gain négatif de -95 % à -240 % pour TACH). **Ce résultat n'est pas lissé ni corrigé** : il indique que le benchlearning international, tel qu'opérationnalisé ici, apporte une valeur ajoutée réelle pour les cibles cycliques mais pas pour les cibles structurelles — cohérent avec l'avertissement du cadrage selon lequel les trajectoires étrangères ne sont pas mécaniquement transposables (Ch.3.1-3.3).

## 7. Vérifications de cohérence (explicitement demandées)

`src/etape3/combination.py::sanity_checks`, table `e3_34_verifications_coherence.csv` — 5/5 réussies :
1. `α=0` ⇒ `X_s = X_tend_MAR` exactement.
2. `α=1` ⇒ `X_s = X_bench-comb` exactement.
3. Portefeuille à un seul donneur ⇒ le benchmark combiné reproduit exactement la trajectoire de ce donneur.
4. Poids ne sommant pas à 1 ⇒ rejeté (`ValueError`).
5. Poids négatif ⇒ rejeté (`ValueError`).

## 8. Cohérence macroéconomique : constat, pas de correction silencieuse

La combinaison est appliquée cible par cible, comme le prescrit le cadrage — **elle ne garantit aucune cohérence comptable inter-cibles**. Exemple observé et conservé tel quel : à l'horizon 10 (2034), la tendance nationale de `DETTE_PUBLIQUE` atteint ≈51 % du PIB tandis que le benchmark combiné (tiré vers le bas par le Danemark, très peu endetté) tombe à ≈20 % — un scénario `α` élevé impliquerait une trajectoire de désendettement du Maroc rapide et non étayée par une dynamique budgétaire cohérente (SOBG, croissance) dans le même scénario. **Aucune réconciliation comptable inter-cibles n'a été appliquée** : l'écart est rapporté, pas corrigé, conformément à la consigne.
