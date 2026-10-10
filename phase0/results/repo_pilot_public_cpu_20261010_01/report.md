# Pilote descriptif SymPy sur dépôt réel

Trois réparations historiques présélectionnées ; assertions de chaînes seulement, aucun score SWE-bench officiel ni couverture PASS_TO_PASS.

L'énergie et le coût ci-dessous sont ESTIMÉS, pas mesurés ; ils couvrent le temps mur séquentiel agent + vérification. Les appels distants demandent une hypothèse de puissance agrégée appropriée.

Hypothèses : 100 W, 0.25 €/kWh.

| stratégie | réussites | Wilson 95 % | jetons réels | plafonds réservés | s/tâche | Wh/tâche estimés | €/tâche estimés |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0/3 | [0.000, 0.561] | 2979 | 36864 | 162.448972 | 4.512471435 | 0.001128117859 |

Les plafonds réservés sont des réservations de budget, ni consommation de jetons ni facture.

Dérivation publique distincte : instantanés inchangés remplacés par références épinglées ; chemins home masqués dans les champs textuels. Les commandes ne sont pas un rejeu textuel exact.
