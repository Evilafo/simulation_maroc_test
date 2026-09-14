# Dictionnaire des variables — Étape 1, UC-S1

Sources croisées : `variable_registry.yaml` (registre canonique v1.1.0, benchlearning UC-S1 v1.2.5) et le chapitre 5 du cadrage (`rapport_cadrage_UC-S1_final.pdf`, pages 35-38). Colonnes réellement présentes vérifiées sur `base_MEF_1991_2024_9pays_28indicateurs.csv` (306 lignes, 28 indicateurs) et `base_B_revised_wide.csv` (405 lignes, 30 indicateurs).

## 1. Cibles officielles UC-S1 (cadrage Ch. 5.1, tableau 5.1)

| Code cadrage | Cible | Colonne réelle | Présente fichier de travail (28 ind.) | Présente base source (30 ind.) | Registre : `uc_s1_target` | Remarques |
|---|---|---|---|---|---|---|
| PIBR | Croissance du PIB réel | `CROISSANCE_PIB` | Oui | Oui | true | RAS |
| INFL | Inflation totale **et sous-jacente** | `INFLATION_CPI` | Oui | Oui | true | Seule l'inflation **totale** est disponible ; aucune colonne d'inflation sous-jacente n'existe dans les deux bases. Écart cadrage → donnée, non comblé. |
| SOBG | Solde budgétaire global **et primaire** | `SOBG` (+ `SOPR` en complément) | Oui (24 manquants sur 306) | Oui | true | `SOPR` présent et documenté comme complément, conformément au cadrage. |
| DETP | Dette publique / PIB | `DETTE_PUBLIQUE` | **Non** | Oui | true | Cible officielle au sens du registre, disponible dans la base source (périmètre APU, `IMF GGXWDG_NGDP`), mais absente du fichier de travail 28 indicateurs. Décision non documentée dans le registre ni le cadrage — voir `MATRICE_CONFORMITE.md`. |
| BCUR | Solde du compte courant | `BALANCE_COURANTE` | Oui | Oui | true | RAS |
| CHOM | Chômage **et élasticité emploi-croissance** | `TACH` | Oui | Oui | true | Le taux de chômage seul est disponible. La variable `ELASTICITE_EMPLOI` a été **retirée** du registre canonique (audit_ref P0-04 : « utilise le taux d'emploi au lieu de l'effectif, à reconstruire avec effectif ») — non reconstruite à ce stade, donc l'élasticité emploi-croissance n'est actuellement pas modélisable avec ces bases. |
| FBCF | FBCF / PIB | `FBCF` | Oui | Oui | true | RAS |

## 2. Variable exclue du périmètre du fichier de travail mais présente dans la base source

| Variable | Rôle possible (cadrage) | Présente fichier de travail | Présente base source | Remarque |
|---|---|---|---|---|
| `TCER` (taux de change effectif réel) | Explicative de BALANCE_COURANTE (Ch. 5.2.5 : « compétitivité et demande extérieure ») | **Non** | Oui | Disponible dans le registre (source BAM/WB, `PX.REX.REER`) et dans `base_B_revised_wide.csv`. Absente du fichier de travail 28 indicateurs sans justification retrouvée. |

## 3. Variables explicatives potentielles présentes dans le fichier de travail (21 variables hors cibles)

| Variable | Label registre | Catégorie | `variable_type` (registre, pilote la politique d'imputation) |
|---|---|---|---|
| `DETO` | Dépenses publiques totales | Finances publiques | ratio_smooth |
| `RETO` | Recettes des administrations publiques | Finances publiques | ratio_smooth |
| `SOPR` | Solde primaire des administrations publiques | Finances publiques | cyclical |
| `EXPO` | Exportations de biens et services | Commerce extérieur | ratio_smooth |
| `IMPO` | Importations de biens et services | Commerce extérieur | ratio_smooth |
| `OUVERTURE_COM` | Ouverture commerciale (BM canonique) | Commerce extérieur | ratio_smooth |
| `EXPORT_MARCHANDISES_USD` | Exportations de marchandises (USD courants) | Commerce extérieur | monetary_flow |
| `TAAC` | Taux d'activité total, 15-64 ans | Marché du travail | ratio_smooth |
| `TAUX_EMPLOI_15P` | Taux d'emploi, 15 ans et plus | Marché du travail | ratio_smooth |
| `POP_ACTIVE_TOTALE` | Population active totale (effectif) | Marché du travail | level_slow |
| `CRDE` | Croissance démographique | Démographie | ratio_smooth |
| `VAAG`, `VAIN`, `VASE` | Valeur ajoutée agriculture / industrie / services | Structure économique | ratio_smooth |
| `EMAG`, `EMIN`, `EMSE` | Emploi agriculture / industrie / services (OIT modélisé) | Structure économique | ratio_smooth |
| `DEP_SANTE_PUBLIQUE` | Dépenses publiques domestiques de santé | Développement humain | ratio_smooth |
| `EVNA` | Espérance de vie à la naissance | Développement humain | level_slow |
| `IDEE` | IDE, entrées nettes | Investissement | cyclical |
| `LFPRF` | Taux d'activité féminin | Genre | ratio_smooth |
| `UNFE` | Chômage féminin | Genre | ratio_smooth |
| `WSWF` | Salariées parmi l'emploi féminin | Genre | ratio_smooth |

Variables du registre **absentes** du fichier de travail sans lien avec DETP/TCER : `COEM`, `SUBV`, `MARGE_BUDGETAIRE` (dérivée, exclusion cohérente avec le blocage leakage documenté par l'audit de modélisabilité antérieur), `TERMES_ECHANGE_NETS`, `CRPA` (dérivée), `TSSE`, `TSSU`, `FBCF_PRIVEE`, `CRSP`, `PAUVRETE_ECO`, `GINI` (ces deux dernières marquées `survey_based`, non imputables par construction). Liste complète : `outputs/tables/06_registre_vs_fichier_travail.csv`.

## 4. Identités comptables/compositionnelles exactes détectées dans le fichier de travail

Voir `outputs/tables/07_identites_lineaires_exactes.csv` — vérifiées empiriquement, pas supposées :

- **`SOBG = RETO − DETO`** (exact à la précision machine, n=282 lignes testables) — ne jamais utiliser `RETO` et `DETO` simultanément comme prédicteurs de `SOBG` à l'étape 2 (identité comptable, pas relation statistique).
- **`OUVERTURE_COM = EXPO + IMPO`** (exact) — ne pas combiner les trois dans un même modèle linéaire.
- **`EMAG + EMIN + EMSE = 100`** (exact à 10⁻⁴ près) — parts d'emploi sectoriel compositionnelles, n'en retenir que 2 sur 3 comme prédicteurs indépendants.
- `VAAG + VAIN + VASE` ≈ 91 % en moyenne (écart-type 5,4 points) : **pas** une identité exacte, colinéarité forte mais non totale.

Ces trois identités exactes expliquent le rang déficient observé sur l'ACP2 (28 variables mais seulement 25 valeurs propres non nulles, cf. `outputs/tables/31_acp2_valeurs_propres.csv`).

## 5. Périmètre de la dette publique (clarification pour l'étape 2)

Le registre confirme que `DETTE_PUBLIQUE` est mesurée au périmètre **administrations publiques (APU)**, code IMF `GGXWDG_NGDP`, avec le code Banque mondiale `GC.DOD.TOTL.GD.ZS` (administration centrale) en repli explicitement documenté comme **non substituable silencieusement** (`perimeter_note` du registre). Il n'y a donc pas de blocage de disponibilité au niveau de la base source : l'absence de `DETTE_PUBLIQUE` dans le fichier de travail 28 indicateurs est un choix de périmètre de ce fichier précis, pas une contrainte de donnée.
