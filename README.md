# DATALAB MEF Maroc — UC-S1 (projet complet, étapes 1-3)

Sous-projet du dépôt `jarvis-starter-kit`, livrant l'intégralité de la mission UC-S1 : audit et ACP (étape 1), caractéristiques et modèles de prévision (étape 2), benchlearning et simulation de trajectoires (étape 3). Les données sources restent sur `Desktop/Work_DAVID_DG_Mohamed/` (chemins absolus dans `configs/config.yaml`) : elles ne sont ni copiées ni modifiées par ce projet.

**Démarrage rapide — pour comprendre les résultats sans lire le code :** `docs/SYNTHESE_METIER_FINALE.md`.

## Structure

```
mef_maroc_ucs1/
├── configs/
│   ├── config.yaml            # etape 1 : chemins, pays, periodes, graines, ACP/clustering
│   ├── config_etape2.yaml     # etape 2 : protocole temporel, cibles, caracteristiques, modeles, evaluation
│   └── config_etape3.yaml     # etape 3 : donneurs, poids, scenarios alpha, Monte Carlo -- valide au demarrage
├── src/
│   ├── io_utils.py, audit.py, pca_core.py, pca_profiles.py, pca_trajectories.py, clustering.py, plotting.py   # etape 1
│   ├── etape2/    # data_extended, temporal, features, selection, stationarity, cointegration_var, baselines, models, evaluation
│   └── etape3/    # config_check, donors, weights, trend_projection, combination, montecarlo
├── run_etape1.py                              # etape 1
├── run_etape2_part{1,2,3}_*.py                # etape 2
├── run_etape3_part{1,2,3,4,5}_*.py            # etape 3 (part5 accepte --fast pour un mode rapide, non livrable final)
├── notebooks/01_etape1_*.ipynb, 02_etape2_*.ipynb, 03_etape3_*.ipynb   # notebooks maitres, executes depuis un noyau propre
├── tests/    # test_data_integrity, test_etape2_integrity, test_etape3_integrity (27 tests au total)
├── outputs/{tables,figures}/, outputs/etape2/{tables,figures,models}/, outputs/etape3/{tables,figures,models}/
└── docs/     # dictionnaire, matrice de conformite (3 etapes), syntheses par etape, synthese metier finale
```

## API FastAPI

L'API charge les artefacts Elastic Net finaux de l'etape 2 et reconstruit les
caracteristiques causales a partir d'un historique annuel fourni par le client.
Elle expose `GET /health`, `GET /models` et `POST /predict`.

```bash
python3 -m pip install -r requirements-api.txt
python3 -m uvicorn API.api:app --reload
```

La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`.
Les fichiers pickle sont des artefacts locaux de confiance et doivent etre
reconstruits si la version de scikit-learn ou les donnees changent.

Pour lancer l'interface graphique dans un second terminal :

```bash
python3 -m http.server 5500 --directory interface
```

Puis ouvrir `http://127.0.0.1:5500` dans le navigateur, avec l'API déjà
démarrée sur le port 8000.

## Docker Compose

Les deux composants peuvent aussi être lancés dans des conteneurs séparés :

```bash
docker compose up --build
```

L'interface est alors disponible sur `http://127.0.0.1:8080` et l'API sur
`http://127.0.0.1:8000/docs`. L'image API embarque les modèles sauvegardés dans
`outputs/etape2/models`, ainsi que le code de reconstruction des caractéristiques.
Pour arrêter les conteneurs :

```bash
docker compose down
```

## Exécution (Windows / VS Code / Jupyter)

Environnement utilisé : Anaconda3 (Python 3.10.9), `C:\Users\MR KITOHOU\anaconda3\python.exe`. Ni `python`/`py`/`conda` ni `jupyter` ne sont sur le PATH par défaut de ce terminal : invoquer les exécutables par leur chemin complet, ou activer l'environnement au préalable (`C:\Users\MR KITOHOU\anaconda3\Scripts\activate.bat`). Aucune dépendance hors distribution Anaconda3 standard (pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, pyyaml, nbformat/nbclient pour les notebooks) — si l'environnement cible ne dispose pas de l'une d'elles, créer un environnement dédié (`conda create -n ucs1 python=3.10 pandas numpy scikit-learn scipy statsmodels matplotlib pyyaml nbformat nbclient ipykernel`) plutôt que de supposer leur présence.

```powershell
# Depuis mef_maroc_ucs1/ -- ordre obligatoire (chaque etape/partie lit les sorties de la precedente)
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape1.py

& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape2_part1_audit_var.py
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape2_part2_models.py
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape2_part3_synthese.py

& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part1_donors_weights.py
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part2_trend_projections.py     # ~6-7 minutes (630 modeles)
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part3_backtest_combination.py  # ~2 minutes
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part4_combination_scenarios.py
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part5_montecarlo.py            # mode complet, 10000 tirages -- LIVRAISON FINALE
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part5_montecarlo.py --fast     # mode rapide (500 tirages), developpement uniquement
& "C:\Users\MR KITOHOU\anaconda3\python.exe" run_etape3_part6_figures.py               # figures MEF (francais), donnees exportables

# Tests (27 au total, quelques secondes)
& "C:\Users\MR KITOHOU\anaconda3\python.exe" -m pytest tests\ -q

# Notebooks (dans VS Code : selectionner le kernel "python3" d'Anaconda3)
```

## Résultats clés par étape

- **Étape 1** (`docs/AUDIT_ET_ACP_ETAPE1.md`) : audit des données, deux ACP (profils moyens / trajectoires), typologie k=2/3. Le Maroc se regroupe avec la Turquie sur l'axe institutionnel/chômage, pas avec les comparateurs principaux du cadrage.
- **Étape 2** (`docs/ETAPE2_PROTOCOLE_ET_CARACTERISTIQUES.md`, `ETAPE2_MODELES_ET_VALIDATION.md`) : aucun VAR/VECM admissible sur le Maroc seul (34 ans de données) ; portefeuille baselines/Elastic Net/panel évalué honnêtement (14/15 cibles en variation battent le seuil de gain de 10 %, 13/20 cibles en niveau/ratio battent leur baseline).
- **Étape 3** (`docs/ETAPE3_BENCHLEARNING_METHODE.md`, `ETAPE3_RESULTATS_SIMULATION.md`) : équations 3.14-3.15 du cadrage implémentées littéralement, S_POC exploratoire distinct du score complet, poids statistiques/politiques strictement séparés, 630 projections tendancielles 2025-2034, Monte Carlo à 10 000 tirages. **Constat central, testé et non lissé : le benchlearning international bat la persistance simple sur la croissance et l'inflation, mais fait moins bien sur les 4 autres cibles.**

Toutes les décisions et écarts documentés (cadrage vs demande vs proposition méthodologique) : `docs/MATRICE_CONFORMITE.md` (43 lignes couvrant les 3 étapes). Synthèse pour décideurs non techniques : `docs/SYNTHESE_METIER_FINALE.md`.

## Limites majeures assumées (voir la matrice de conformité pour le détail complet)

BVAR, Boosting/MLP, DFM/MIDAS et SVAR ne sont pas implémentés (moteurs 2 à 4 et module structurel du cadrage Ch.6.2) ; le score d'éligibilité complet à 7 critères (S1 PIB/habitant PPA, S4 ressources, S5 qualité institutionnelle WGI) n'est pas reconstituable avec les données disponibles ; les horizons de projection 6 à 10 ans (2030-2034) sont des extrapolations non testées en backtest ; la couverture empirique des intervalles de confiance n'est pas vérifiable finement avec 4 réalisations par horizon.
# simulation_maroc_test
# simulation_maroc_test
