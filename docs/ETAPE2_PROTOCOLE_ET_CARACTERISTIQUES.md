# Étape 2 — Protocole temporel, caractéristiques, stationnarité et cointégration

Statut global de cette partie : **implémenté, exécuté, évalué**. Aucune validation métier (DEPF/DTFE) à ce stade.

## 1. Jeu de travail étendu (30 indicateurs)

Décision actée (met à jour le point ouvert #1 de `docs/SUIVI_MODELISATION.md` issu de l'étape 1) : `DETTE_PUBLIQUE` et `TCER` sont réintégrés depuis `base_B_revised_wide.csv`, filtré sur 1991-2024/9 pays. Vérifié automatiquement (`tests/test_etape2_integrity.py::test_extended_matches_etape1_on_shared_columns`) : les 28 colonnes partagées avec le fichier de travail de l'étape 1 sont identiques à 10⁻⁶ près. Le Maroc dispose d'une couverture **complète** (0 manquant/34) sur les deux variables réintégrées ; la panel comparateurs présente des lacunes documentées (`configs/config_etape2.yaml::reintegrated_variables`), notamment `TCER` totalement indisponible pour la Turquie et le Viet Nam.

## 2. Protocole temporel

- **Prévision directe par horizon** (pas de prévision itérée pour Elastic Net/panel) : un jeu d'entraînement distinct `(X_o, y_{o+h})` par horizon `h`, réajusté à chaque origine.
- **Règle de non-fuite stricte** : pour produire une prévision à l'origine `O` et l'horizon `h`, seules les paires `(o', y_{o'+h})` dont l'étiquette est déjà connue à `O` (c'est-à-dire `o' ≤ O-h`) sont utilisables à l'entraînement — pas seulement `o' < O`. Testé explicitement (`test_training_rows_never_include_future_labels`).
- **Gel des choix** : caractéristiques (après filtrage de redondance) et hyperparamètres Elastic Net (α, l1_ratio) sont choisis **une seule fois** par validation croisée chronologique sur le bloc interne (`target_year ≤ 2020`), puis gelés. Chaque origine (interne ou finale) ne fait que réajuster les coefficients sur les données strictement antérieures — jamais de nouveau réglage sur le bloc final 2021-2024.
- **Bloc final** : ce ne sont pas les origines 2021-2024, mais les prévisions dont **l'année cible** tombe dans [2021, 2024]. Pour h=5, l'origine correspondante (2019) est antérieure à 2020 : conformément à la consigne, cette origine est utilisée et l'effectif par horizon est rapporté (4 prévisions finales pour chaque horizon 1 à 5, cf. `outputs/etape2/tables/e2_11_evaluation_vs_baseline.csv`, colonne `n_previsions`).
- **Millésimes** : rappel documenté en étape 1, toujours valable — la base n'a pas de vintages réels. Le backtest est qualifié de **rétrospectif sur données révisées**, pas de pseudo-temps-réel. Hypothèse retenue : l'information de l'année O est supposée disponible en bloc à l'origine O (données annuelles uniquement).

## 3. Matrice cible → explicatives (cadrage Ch. 5 → colonnes disponibles)

| Cible | Mécanismes du cadrage disponibles (colonnes utilisées) | Mécanismes du cadrage **non disponibles** dans cette base |
|---|---|---|
| CROISSANCE_PIB | FBCF (investissement), IDEE (IDE), TCER (environnement extérieur), autorégressif | Crédit au secteur privé (CRSP absent), capital humain (TSSE/TSSU absents), croissance zone euro |
| INFLATION_CPI | TCER (inflation importée, proxy), CROISSANCE_PIB (pression de demande, proxy), autorégressif | Output gap explicite, M3, prix du pétrole, salaires |
| SOBG | CROISSANCE_PIB (effet cycle), DETTE_PUBLIQUE (charge de la dette), autorégressif | RETO/DETO/SOPR **exclus** : composants algébriques exacts ou quasi-exacts de SOBG (identité `SOBG=RETO-DETO` vérifiée en étape 1 ; SOPR r=0.75 avec SOBG) — fuite garantie si inclus |
| DETTE_PUBLIQUE | SOPR (solde primaire), CROISSANCE_PIB (effet taux-croissance, proxy), TCER (risque de change, proxy), autorégressif | Taux d'intérêt effectif de la dette, passifs éventuels |
| BALANCE_COURANTE | TCER (compétitivité), IDEE (financement externe), autorégressif | Termes de l'échange nets (retirés du fichier de travail), croissance des partenaires, transferts MRE/tourisme |
| TACH | CROISSANCE_PIB (loi d'Okun), TAAC + POP_ACTIVE_TOTALE (offre de travail), EMAG + EMIN (structure sectorielle — 2 des 3 parts pour éviter la colinéarité compositionnelle exacte EMAG+EMIN+EMSE=100), autorégressif | Salaire minimum réel, dépenses de politiques actives de l'emploi, capital humain |
| FBCF | IDEE (investissements complémentaires), CROISSANCE_PIB (anticipations, proxy), autorégressif | Crédit au secteur privé, profits des entreprises, indice d'incertitude |

Détail machine-lisible : `configs/config_etape2.yaml::target_mechanisms`. Caractéristiques dérivées effectivement générées par colonne : niveau, retards 1 à 3, différence première, moyenne/écart-type mobiles (fenêtres 3 et 5, strictement passées), logarithme et log-différence si la série est strictement positive. Plafond de 6 caractéristiques finales par spécification via la sélection Elastic Net (parcimonie).

**Cibles non modélisables, avec motif (aucune non déclarée n'est inventée)** :
- *Élasticité emploi-croissance* (volet du cadrage CHOM) : variable retirée du registre canonique (audit v1.1, formule invalide), non reconstruite.
- *Inflation sous-jacente* (volet du cadrage INFL) : aucune colonne correspondante dans la base source.

## 4. Sélection à trois niveaux

1. **Plausibilité/disponibilité** : matrice ci-dessus (imposée en amont, avant tout calcul).
2. **Filtrage des redondances** : élimination des caractéristiques dont la corrélation absolue dépasse 0,90 avec une caractéristique déjà retenue (calculé sur le bloc interne uniquement, jamais sur le bloc final), en conservant en priorité les variables les plus interprétables (ordre de `target_mechanisms`).
3. **Sélection supervisée parcimonieuse** : Elastic Net réglé par validation croisée chronologique (walk-forward), jamais par *k-fold* aléatoire — grille `α ∈ {0.01,…,3.0}`, `l1_ratio ∈ {0.1,…,0.9}` (`configs/config_etape2.yaml::selection`). Pour le panel pooled, le découpage de validation est fait **par année réelle** (pas par position dans un index concaténé) pour respecter la chronologie malgré le regroupement de plusieurs pays — bug corrigé en cours de développement, testé.

Importance par permutation et SHAP : **non implémentés à ce stade** (limite explicitement assumée, faute de temps disponible dans cette passe ; les coefficients Elastic Net standardisés, déjà exportés, donnent une lecture de direction/magnitude relative suffisante pour ce premier tour).

## 5. Stationnarité (Maroc uniquement, ADF + KPSS)

Résultats complets : `outputs/etape2/tables/e2_01_stationnarite_maroc.csv`. Règle de classification combinée standard (ADF H0 racine unitaire, KPSS H0 stationnarité), jamais forcée vers I(0)/I(1) en cas de désaccord :

| Résultat | Variables |
|---|---|
| **I(0)** (concordant) | CROISSANCE_PIB, INFLATION_CPI |
| **I(1)** (concordant) | DETO, EMIN, FBCF, RETO, TAAC, TACH |
| **Indéterminé** (ADF et KPSS ne rejettent aucun des deux H0 — puissance faible probable avec N=34) | BALANCE_COURANTE, DETTE_PUBLIQUE, IDEE, TCER, POP_ACTIVE_TOTALE |
| **Schéma non standard** (à examiner manuellement, pas assimilé par défaut) | SOBG, SOPR, EMAG |

**Aucune série n'est forcée vers un ordre d'intégration commode.** Les cibles budgétaires (SOBG) et la dette (DETTE_PUBLIQUE) restent d'un statut incertain avec cette taille d'échantillon — cohérent avec l'audit de modélisabilité antérieur.

## 6. Cointégration et admissibilité VAR/VECM

Quatre systèmes candidats testés (`configs/config_etape2.yaml::var_candidate_systems`, `outputs/etape2/tables/e2_02_systemes_var_vecm.csv`) :

| Système | Variables | Ordres d'intégration | Décision |
|---|---|---|---|
| dette_solde | DETTE_PUBLIQUE, SOBG | indéterminé, indéterminé | **Réexamen** |
| solde_compte_courant | SOBG, BALANCE_COURANTE | indéterminé, indéterminé | **Réexamen** |
| croissance_investissement | CROISSANCE_PIB, FBCF | I(0), I(1) | **Réexamen** (mélange I(0)/I(1)) |
| compte_courant_change | BALANCE_COURANTE, TCER | indéterminé, indéterminé | **Réexamen** |

**Aucun des quatre systèmes candidats n'atteint le statut « tous I(1) confirmés »** requis avant même de calculer un rang de Johansen : la règle de décision (section 3 de la demande) est appliquée strictement, sans VECM mécanique. Ce résultat est cohérent avec le verdict de l'audit de modélisabilité antérieur (`context/audit-modelisabilite-uc-s1/06_VERDICT_ARCHITECTURE.md`) qui proscrivait déjà VAR/VECM sur le Maroc seul à ce stade.

**Repli retenu** : pour le système `croissance_investissement` (mélange I(0)/I(1), cas le plus proche d'un ARDL/ECM), la dynamique retard/différence entre `CROISSANCE_PIB` et `FBCF` est déjà couverte par la spécification Elastic Net de `CROISSANCE_PIB` (qui inclut les retards de `FBCF`) — jouant le rôle du repli ARDL/ECM prescrit par la demande, sans construction d'un module de test de bornes séparé (limite de portée assumée, documentée ici plutôt que dissimulée).

**Aucun VAR/VECM n'a donc été estimé sur le Maroc dans cette étape** : les quatre `motif_decision` ci-dessus constituent le rejet consigné exigé par la demande. Le portefeuille prédictif (section suivante) repose sur les baselines, Elastic Net et le panel — conformément à l'architecture « Piste A » déjà recommandée par l'audit antérieur.
