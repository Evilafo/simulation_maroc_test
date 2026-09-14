# DATALAB MEF Maroc — UC-S1 : synthèse métier

*Document de synthèse pour les décideurs du Ministère de l'Économie et des Finances. Détails techniques complets dans les documents référencés en fin de note.*

## Ce que ce travail apporte

Une architecture reproductible pour situer le Maroc parmi 8 pays de comparaison (Chili, Costa Rica, Danemark, Malaisie, Portugal, Turquie, Uruguay, Viêt Nam) sur 7 indicateurs macroéconomiques clés, et pour construire des trajectoires de référence à horizon 2025-2034 combinant la dynamique propre du Maroc et l'expérience de ces pays. L'ensemble repose sur des données réelles (1991-2024), des méthodes auditées, et des résultats testés a posteriori sur la période récente (2021-2024) avant d'être présentés comme utilisables.

## Ce qui différencie le Maroc dans l'échantillon

L'analyse en composantes principales (étape 1) situe le Maroc près de la moyenne de l'échantillon en matière d'ouverture commerciale, mais nettement à l'écart sur l'axe institutionnel/emploi : le Maroc se regroupe avec la **Turquie**, pas avec le Viêt Nam ni le Costa Rica pourtant retenus comme comparateurs principaux dans le cadrage. Le point commun identifié est un **taux de chômage structurellement élevé**, pas un retard de développement général. Ce n'est pas une critique du choix des comparateurs (fondé sur d'autres critères — niveau initial de développement, structure productive, disponibilité des données), mais un rappel qu'une proximité statistique ne dit pas tout.

## Apport réel des comparateurs internationaux (constat testé, pas supposé)

Le test le plus important de cette mission : est-ce que combiner les trajectoires de Viêt Nam, Costa Rica et Danemark améliore réellement la prévision de l'évolution du Maroc ? Réponse, vérifiée sur les 4 dernières années réelles (2021-2024) :

- **Oui, nettement, pour la croissance du PIB et l'inflation** — la combinaison des comparateurs bat une simple projection de la tendance récente de 27 à 61 % selon l'horizon.
- **Non, pour la dette publique, le solde budgétaire, le compte courant et le chômage** — sur ces indicateurs plus structurels, la combinaison internationale fait **moins bien** qu'une hypothèse simple de prolongation de la dernière valeur connue.

Interprétation : les dynamiques conjoncturelles (croissance, prix) sont suffisamment partagées à l'échelle internationale pour que l'apprentissage croisé apporte de la valeur. Les dynamiques structurelles (finances publiques, marché du travail) sont trop spécifiques à chaque pays pour être bien captées par une moyenne pondérée d'expériences étrangères, au moins avec la méthode et les données mobilisées ici.

## Résultats par cible (résumé)

| Cible | Modèle propre au Maroc (étape 2) | Apport des comparateurs (étape 3) | Statut |
|---|---|---|---|
| Croissance du PIB | Bat la marche aléatoire de 35 à 67 % selon l'horizon | Bat aussi la marche aléatoire (27-61 %) | Résultat le plus solide du portefeuille |
| Inflation | Bat la marche aléatoire de 12 à 36 % | Bat aussi la marche aléatoire (5-38 %) | Solide |
| Chômage | Résultats mitigés (gain sur 4 horizons sur 5) | Nettement moins bon que la persistance simple | Modèle national à privilégier |
| Solde budgétaire | Résultats mitigés, cible la plus exposée aux biais comptables | Moins bon que la persistance simple | Prudence requise |
| Dette publique / PIB | La cible la plus difficile à prévoir du portefeuille, sur les deux approches | Idem, écart encore plus marqué aux horizons longs | Aucune méthode testée ici n'est pleinement satisfaisante |
| Compte courant / PIB | Résultats mitigés selon l'horizon | Nettement moins bon que la persistance simple | Modèle national à privilégier |
| FBCF / PIB | Résultats mitigés selon l'horizon | Non testé au même niveau de détail (limite de portée) | Résultats partiels |

**Aucun résultat ci-dessus n'est présenté comme définitivement validé** : le nombre d'années de test réel (4, pour 2021-2024) reste trop faible pour une confirmation statistique fine. Ce sont des indications solides, pas des certitudes.

## Limites à connaître avant tout usage opérationnel

1. **Aucun modèle VAR/VECM n'a pu être estimé sur le Maroc seul** : le volume de données annuelles disponibles (34 ans) reste sous les seuils usuels de fiabilité pour ce type de modèle multivarié. Les résultats reposent sur des méthodes plus sobres (régression régularisée, comparaisons de panel).
2. **Les projections au-delà de 2029 (horizons 6 à 10 ans) sont des extrapolations non testées** — aucune donnée réelle ne permet de vérifier leur fiabilité à ce stade. À traiter avec prudence accrue.
3. **La combinaison entre la tendance marocaine et les comparateurs ne garantit pas la cohérence entre indicateurs** : par exemple, un scénario fortement orienté vers les comparateurs peut impliquer un désendettement rapide de la dette publique sans que le solde budgétaire du même scénario ne le permette. Ces écarts sont documentés, pas corrigés automatiquement.
4. **Deux décisions restent à confirmer avec les équipes DEPF/DTFE** : le choix de la fenêtre 1991-2024 plutôt qu'une autre fenêtre du registre de données, et l'exemplaire exact du rapport de cadrage à utiliser comme référence (un doublon de nom de fichier n'a pas pu être localisé avec certitude).
5. **Deux indicateurs prévus au cadrage restent non disponibles** : l'élasticité emploi-croissance (formule invalidée lors d'un audit antérieur, non reconstruite) et l'inflation sous-jacente (absente des bases de données mobilisées).

## Données complémentaires prioritaires pour améliorer ce dispositif

Par ordre d'impact attendu :
1. **Séries infra-annuelles marocaines** (mensuelles/trimestrielles : HCP, TGR, Bank Al-Maghrib) — permettraient le nowcasting prévu par le cadrage, non réalisable avec les seules données annuelles mobilisées ici.
2. **Indicateurs institutionnels et de niveau de vie** (PIB/habitant en parité de pouvoir d'achat, indicateurs de gouvernance WGI) — nécessaires pour appliquer la grille de sélection des pays comparateurs du cadrage dans sa version complète (actuellement, seule une version partielle a pu être calculée).
3. **Historique des révisions de données (millésimes)** — la base actuelle ne conserve qu'une version par série ; toute évaluation de performance reste donc rétrospective sur données déjà révisées, pas une simulation de conditions réelles de prévision.
4. **Reconstruction de l'effectif de population active** pour recalculer une élasticité emploi-croissance valide.

## Documents de référence

- Étape 1 (audit, ACP) : `docs/AUDIT_ET_ACP_ETAPE1.md`
- Étape 2 (modèles, validation) : `docs/ETAPE2_PROTOCOLE_ET_CARACTERISTIQUES.md`, `docs/ETAPE2_MODELES_ET_VALIDATION.md`
- Étape 3 (benchlearning, simulation) : `docs/ETAPE3_BENCHLEARNING_METHODE.md`, `docs/ETAPE3_RESULTATS_SIMULATION.md`
- Toutes les décisions et écarts documentés : `docs/MATRICE_CONFORMITE.md`
- Point d'entrée technique : `README.md`
