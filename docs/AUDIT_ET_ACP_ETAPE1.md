# Audit et ACP — synthèse de l'Étape 1

Statut global : **exécuté et évalué**. Aucune étape n'est **validée** au sens métier (pas de revue DEPF/DTFE à ce stade). Rien n'est **bloqué** de façon irréversible ; un point est **bloqué en l'état** (élasticité emploi-croissance, voir §1.3).

## 1. Audit des données

### 1.1 Conformité aux chiffres annoncés

Tous les chiffres de la demande sont vérifiés exactement sur les fichiers réels :
- `base_B_revised_wide.csv` : 405 lignes (9 pays × 45 années 1980-2024), 30 indicateurs. Deux copies identiques trouvées sur la machine (hash SHA256 identique) ; celle de `Data_collection/data/` est retenue comme chemin canonique.
- `base_MEF_1991_2024_9pays_28indicateurs.csv` : 306 lignes (9 × 34), 28 indicateurs, 101 cellules manquantes, 278 états complets / 28 incomplets, 0 doublon pays-année, aucune année manquante par pays.
- Exclusions `DETTE_PUBLIQUE` et `TCER` confirmées.

### 1.2 Structure des valeurs manquantes

Aucune lacune interne ni de fin de série : uniquement des débuts de série tardifs, concentrés sur des pays de comparaison, jamais sur le Maroc :

| Variable(s) | Pays concernés | Période manquante |
|---|---|---|
| SOBG, SOPR, RETO | Turquie, Uruguay, Viet Nam | 1991-1999 (fin variable selon le pays) |
| DETO | Turquie (1 an), Viet Nam | 1991-1999 |
| INFLATION_CPI, FBCF | Viet Nam | 1991-1994/1995 |
| VAAG, VAIN, VASE | Portugal | 1991-1994 |

La fenêtre **2000-2024** est la première fenêtre entièrement sans lacune sur les 9 pays × 28 indicateurs — elle coïncide avec le `window_core` du registre canonique (constaté, non recherché) et sert de fenêtre de sensibilité.

### 1.3 Dictionnaire et cibles

Le mapping cadrage → colonnes fourni dans la demande est confirmé exact à 100 % après lecture directe du chapitre 5 du rapport de cadrage (pages 35-38). Deux nuances documentées :
- **INFL** : seule l'inflation totale (`INFLATION_CPI`) est disponible ; l'inflation sous-jacente mentionnée au cadrage n'existe dans aucune des deux bases.
- **CHOM** : le taux de chômage (`TACH`) est disponible ; l'élasticité emploi-croissance mentionnée au cadrage correspond à une variable (`ELASTICITE_EMPLOI`) **explicitement retirée** du registre canonique lors de l'audit v1.1 (formule jugée invalide : elle utilisait un taux d'emploi au lieu d'un effectif). Elle n'a pas été reconstruite dans le cadre de cette étape — **statut : bloqué**, à traiter en amont de toute modélisation du chômage si cette dimension est requise.

### 1.4 Identités comptables exactes

Trois identités linéaires exactes ont été vérifiées empiriquement (et non supposées) dans le fichier de travail :
1. `SOBG = RETO − DETO` (n=282 lignes testables, écart nul à la précision machine).
2. `OUVERTURE_COM = EXPO + IMPO` (exact).
3. `EMAG + EMIN + EMSE = 100` (exact à 10⁻⁴ près).

Ces trois identités expliquent intégralement le rang déficient observé sur l'ACP2 (25 valeurs propres non nulles sur 28 attendues). **Implication directe pour l'étape 2** : ne jamais combiner ces variables comme prédicteurs simultanés d'une même cible (fuite garantie par construction algébrique, pas par apprentissage).

### 1.5 Score S_POC et Min-Max (clarification)

La lecture du chapitre 3 du cadrage confirme qu'il n'existe **aucune mention d'ACP, de composantes principales ou de classification** dans les chapitres 2 à 6 du rapport de cadrage. Le score de sélection des pays comparateurs y est décrit (grille S1 à S7, section 3.2.2-3.2.5) : S1 = niveau initial de développement (PIB/hab. PPA), S2 = structure productive, S3 = ouverture commerciale, S4 = contraintes en ressources, S5 = qualité institutionnelle (WGI), S6 = profondeur/qualité des données, S7 = pertinence de la trajectoire de transformation. Le score S_POC mentionné dans la consigne (construit sur S2, S3, S6 et une partie de S7) est cohérent avec un sous-score **calculable à partir des seules variables disponibles dans ces bases** (S1 nécessite le PIB/hab. PPA, S4 les indicateurs hydriques/énergétiques, S5 les WGI : aucun des trois n'est présent dans le fichier de travail 28 indicateurs). Les deux ACP de cette étape sont donc bien une **proposition méthodologique distincte** du score S_POC, non un substitut ni une implémentation de celui-ci.

## 2. ACP1 — Profils moyens par pays (1991-2024)

- **F1 (40,9 % de variance)** oppose les économies très ouvertes/exportatrices (Viet Nam, Malaisie — F1 fortement négatif, associées à des niveaux élevés d'exportations, importations et ouverture commerciale) aux économies moins tournées vers l'export sur la période.
- **F2 (24,2 % de variance, 65,1 % cumulé avec F1)** oppose un pôle « développement institutionnel » (Danemark, Portugal — espérance de vie, recettes/dépenses publiques, emploi de services élevés) à un pôle « chômage élevé » (Maroc, Turquie — TACH et chômage féminin fortement chargés négativement).
- **Le Maroc** se situe près de la moyenne sur F1 (structure d'ouverture proche de la moyenne de l'échantillon) mais très chargé négativement sur F2, aux côtés de la Turquie. Un déplacement futur dans ce plan ne serait pas automatiquement un « progrès » : F2 est dominé par le chômage, une baisse du chômage marocain déplacerait le point vers le haut sans résumer à elle seule la trajectoire de développement.
- **Sensibilité fenêtre commune (2000-2024)** : structure de variance expliquée stable, la conclusion n'est pas un artefact des données manquantes en début de période.
- **Sensibilité leave-one-country-out** : F1 reste entre 39,6 % et 45,7 %, F2 entre 19,3 % et 30,0 % selon le pays retiré. Le retrait du Viet Nam réduit le plus F1, celui du Danemark réduit le plus F2 — cohérent avec leur position extrême dans le plan. Avec 9 pays, l'analyse reste exploratoire (rang centré maximal = 8).

## 3. ACP2 — Trajectoires annuelles (1991-2024, 278 états complets)

- F1 (31,7 %) et F2 (21,3 %) reproduisent une structure très proche de celle de l'ACP1 : la lecture openness/chômage-institutions n'est pas un artefact du seul calcul de moyennes par pays.
- Le nuage pays-année forme des paquets bien séparés par pays : l'essentiel de la variance est **entre pays** (structurel), pas **intra-pays au fil du temps** (conjoncturel), sur cette période.
- **Trajectoire du Maroc** : part d'un point très bas sur F2 en 1991 (chômage élevé) et remonte progressivement vers -2,5/-3 en 2024, sans sortir du quadrant « chômage élevé » ni converger visiblement vers le profil d'ouverture du Viet Nam ou de la Malaisie sur F1. Lecture descriptive uniquement : ni la significativité ni la causalité d'une évolution ne sont mesurées à ce stade.

## 4. Typologie des pays

| Espace | k | Partition | Silhouette (K-means / CAH) | Accord K-means/CAH (ARI) | Stabilité (ARI moyen, 20 graines) |
|---|---|---|---|---|---|
| ACP1 (F1-F2) | 2 | {MYS,VNM} vs 7 autres | 0,52 / 0,52 | 1,00 | 1,00 |
| ACP1 (F1-F2) | 3 | {MYS,VNM} / {MAR,TUR} / {CHL,CRI,DNK,PRT,URY} | 0,50 / 0,50 | 1,00 | 1,00 |
| ACP2 (centroïdes) | 2 | Cohérent avec ACP1 | — | — | — |
| ACP2 (centroïdes) | 3 | Désaccord K-means/CAH sur {CHL,CRI} | — | <1,00 | — |

**Point notable** : à k=3, le Maroc se regroupe avec la **Turquie**, pas avec le Viet Nam ni le Costa Rica (les deux « benchmarks principaux » retenus par la grille de sélection du cadrage, chapitre 3). Ceci ne remet pas en cause la sélection du cadrage, fondée sur des critères différents (niveau initial de développement, structure productive, qualité institutionnelle, disponibilité des données) : cela illustre seulement qu'une proximité dans cet espace ACP purement descriptif (28 indicateurs macro-structurels) ne coïncide pas automatiquement avec le score de comparabilité du cadrage. **Une proximité de cluster ne démontre pas la transférabilité d'un benchmark** — aucune conclusion de politique économique n'est tirée de ce seul résultat.

Aucun nombre de classes n'est imposé comme « correct » : k=2 et k=3 sont deux lectures valides à des granularités différentes.

## 5. Limites explicitement non levées à ce stade

- Aucun test statistique de significativité des différences entre groupes.
- La qualité institutionnelle (WGI), le PIB/habitant PPA et les contraintes en ressources (composantes S1/S4/S5 du score de sélection du cadrage) ne sont pas dans les 28 indicateurs : les deux ACP ne peuvent donc pas être comparées terme à terme au score de sélection des pays comparateurs du cadrage.
- Le HTML `Modelisation_Maroc (1).html` (779 Ko) n'a pas été chargé intégralement en conversation (consigne respectée) ; son contenu détaillé n'a pas été audité au-delà de son identification.
