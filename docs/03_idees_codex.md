# Idées de recherche fondées sur les mesures de la phase 0 (Codex, 2026-10-09)

Codex (GPT-6 Astra, effort maximal, lecture seule) a reçu les mesures de la phase 0. Sa consigne : proposer
des idées nouvelles qui tiennent mathématiquement, et les vérifier sur nos fichiers. Ce document en garde
l'essentiel (le rapport brut n'est pas publié). Il
indique ce que j'ai vérifié moi-même et ce que nous en faisons.

Toutes ces analyses portent sur le jeu **test**. Elles sont donc exploratoires : une méthode choisie à
partir d'elles devra être confirmée sur des questions encore inutilisées.

## Le diagnostic : l'essaim n'écoute pas son expert minoritaire

| QCM (test) | fusion | plafond (≥ 1 expert juste) | fusion fausse alors qu'un expert a raison | dont : un seul expert a raison |
| --- | ---: | ---: | ---: | ---: |
| ARC | 84,7 % | 94,5 % | 9,8 % | 73,5 % |
| MMLU-Pro | 31,9 % | 53,7 % | 21,9 % | 77,7 % |

- **L'écart au plafond, c'est un expert seul qui a raison contre les autres.** Quand un seul expert a la
  bonne réponse, la fusion ne le suit que dans 17 % des cas sur ARC et 21 % sur MMLU-Pro.
- **Les familles différentes se trompent ensemble.** La corrélation des erreurs est de 0,37 sur ARC et
  0,40 sur MMLU-Pro. Si les erreurs étaient indépendantes, le plafond serait de 99,7 % sur ARC et 73,9 %
  sur MMLU-Pro, au lieu de 94,5 % et 53,7 %.
- **L'unanimité n'est pas une garantie** : sur MMLU-Pro, les décisions unanimes ne sont justes qu'à 58,8 %.
- **Le « plafond » n'est pas une borne absolue.** C'est la meilleure réponse si l'on savait choisir le bon
  premier choix parmi les experts. La vraie limite d'une décision prise à partir des seules sorties S est
  A*(S) = E[max_c P(Y = c | S)]. Savoir quel expert a raison est une information qui peut être absente
  de S. Il faut donc soit mieux exploiter S, soit **poser une autre question** aux modèles.

**Vérifié par moi** : le plafond (94,47 % et 53,67 %) et la part des décisions où un seul expert a raison
(8,7 % sur ARC, 21,5 % sur MMLU-Pro) sont recalculés indépendamment. Ils sont cohérents avec les chiffres
de Codex.

## Un défaut de notre protocole, confirmé

Sur MMLU-Pro (10 options, 5 passages), `balanced_perm` utilise les décalages 0, 2, 4, 6 et 8. **Une option
paire ne visite que des positions paires, et une option impaire que des positions impaires.** Le graphe
options–positions est coupé en deux (vérifié : 10 nœuds atteignables sur 20). Deux conséquences :

- Un biais de position, par exemple une préférence pour la lettre A, avantage toujours la même moitié
  des options. La moyenne des passages ne l'efface donc pas complètement.
- Le biais de position ne peut pas être séparé du contenu : la matrice est de rang 17 au lieu de 18.

**Correction** : prendre un pas premier avec m, par exemple 3 pour 10 options (décalages 0, 3, 6, 9, 2),
ou remplacer une seule rotation (0, 1, 4, 6, 8). ARC (4 options, pas de 1 en pratique) n'est pas touché.
À appliquer aux prochaines campagnes. Ce changement de protocole fera changer de version les fichiers
MMLU-Pro.

## Les cinq idées, classées

### 1. L'appel du minoritaire : une comparaison à l'aveugle, dans les deux sens

**Le principe.** Quand la 2ᵉ réponse b de la fusion est le premier choix d'au moins un expert, on fait
appel. On retire l'expert qui l'a proposée, et on demande aux trois autres de comparer la réponse a de la
fusion et b, sans leur montrer les votes. On le fait dans les deux ordres d'affichage (a/b puis b/a) pour
annuler le biais d'étiquette : ℓ = (ℓ⁺ + ℓ⁻)/2. On remplace a par b si la médiane des ℓ dépasse un
seuil τ, fixé sur dev.

**Les maths.** Le gain est exactement ΔAcc = B·r − C·h, sans aucune hypothèse d'indépendance, où :

- B = P(appel et b juste), C = P(appel et a juste) ;
- r = le taux de bons renversements, h = le taux de bonnes réponses cassées.

Sur ARC test, B = 8,6 % et C = 30,5 %. **Le cahier des charges** : avec h = 2 %, il faut r > 44 % pour
rattraper Qwen3-4B. Sur MMLU-Pro, il faut r > 64 %. Seuls 54 % des cas récupérables ont la bonne réponse
dans les 2 premiers choix de la fusion, ce qui limite le gain possible.

**Ce que ça coûte.** Un aller-retour de plus, seulement pour les questions qui font appel. La réponse ne
pèse que 24 octets de rapports de vraisemblance.

**Le risque.** Les juges peuvent répéter leur erreur. Retirer le proposant ne rend pas les juges
indépendants pour autant.

**Statut : à tester en premier** (ARC dev, un passage par question, mesurer r et h séparément).

### 2. Le certificat d'arrêt : décider sans attendre les retardataires

**Le principe.** La cible est q = Σᵢ wᵢ pᵢ. Après les réponses d'un ensemble S de pairs, on pose :

- L_c = Σ_{i∈S} wᵢ pᵢ(c) ;
- W = Σ_{i∉S} wᵢ, le poids des pairs qui n'ont pas encore répondu.

Comme une probabilité est toujours entre 0 et 1, L_c ≤ q(c) ≤ L_c + W. La réponse a est donc
**définitivement gagnante** dès que L_a > max_{b≠a} (L_b + W). C'est une preuve immédiate, sans aucune
hypothèse.

**Les chiffres.** Avec 3 experts sur 4, la décision est certifiée exacte dans 75,5 % des cas sur ARC, mais
seulement dans 8,5 % des cas sur MMLU-Pro. On peut ajouter un arrêt *statistique* après 2 experts qui
sont d'accord, en calibrant le risque de s'écarter de la décision complète (Clopper–Pearson).

**Le rejeu exploratoire.** Sur ARC, 61 % des décisions s'arrêtent après 2 réponses (2,78 réponses en
moyenne), sans aucune décision changée sur test.

**Le piège.** Interroger 2 pairs puis les autres si besoin économise du calcul, mais ajoute des
allers-retours. Lancer tout le monde puis décider dès que c'est certain réduit l'attente, sans économiser
de calcul si l'annulation ne marche pas. Il faut mesurer avant de parler de gain de latence.

**Statut : à tester. Facile**, et directement utile au mode k-sur-n de l'expérience 2.

### 3. Des permutations qui permettent de retirer le biais question par question

**Le principe.** Le modèle : log pᵢᵣ꜀ (centré) = θᵢ꜀ (contenu) + βᵢ,pos (position) + bruit, estimé par
moindres carrés, question par question. On fusionne ensuite softmax(θ̂/T). L'identification n'est
possible que si le graphe options–positions est connexe, et la précision dépend de la plus petite valeur
singulière du plan de permutations.

**Ce que ça donne sur nos données.** C'est neutre ou négatif sur ARC (85,3 % contre 86,0 %). Sur MMLU-Pro,
on ne peut pas conclure tant que le défaut de parité n'est pas corrigé.

**Statut : on corrige le plan de permutations** (voir plus haut) ; la correction par question reste une
hypothèse.

### 4. La spéculation exacte avec certificat partiel, sur un alphabet commun d'octets

**Le principe.** La cible est q = Σᵢ wᵢ pᵢ, définie sur les octets, ce qui permet de passer d'un tokenizer
à l'autre. Un brouillonneur propose b ~ d, et on l'accepte si u·d(b) ≤ q(b). C'est le décodage spéculatif
exact. **La nouveauté** est d'accepter dès que u·d(b) ≤ L(b), de rejeter dès que u·d(b) > L(b) + W, et
d'attendre sinon. La décision est exacte quel que soit l'ordre d'arrivée des pairs. Et si le brouillonneur
est l'expert j, tout tirage u ≤ w_j est accepté sans attendre personne.

**Le débit.** Il vaut v = E[A]/E[T] avec α = 1 − TV(d, q). Dans la simulation de Codex (hypothétique),
un bloc trop long *ralentit* : la longueur acceptée plafonne alors que le brouillon coûte toujours.

**Ce que ça coûte.** Il faut la distribution complète pour la correction après un rejet, soit environ
13 Kio par bloc en top-8 avec la masse résiduelle.

**Statut : piste** à fort potentiel pour le réseau étendu, mais difficile : interface en octets, coût de
la vérification.

### 5. Un code correcteur d'erreurs réparti entre les familles

**Le principe.** Chaque option reçoit un mot de 7 bits. Code vérifié : 10 mots, distance minimale 3,
5 « 1 » par colonne. Chaque bit est une question binaire (« la réponse est-elle dans ce groupe ? »)
confiée à une famille, et posée dans les deux sens. Le décodage est sûr tant que 2e + s < 3, où e est le
nombre de bits faux et s le nombre de bits absents : on corrige un bit faux, ou on supporte deux pairs
absents.

**Le risque.** Un modèle qui s'est trompé de réponse dans sa tête répond de façon cohérente avec cette
erreur, et ses bits faux sont corrélés.

**Statut : piste exploratoire.**

### Écarté

- **Pondérer chaque expert par l'entropie et le désaccord entre passages** : 85,7 % contre 86,0 % sur ARC,
  et 37,7 % contre 38,7 % sur MMLU-Pro. Aucun gain.
- **Une diffusion « par consensus » choisie seulement parce qu'elle converge.** Amortir la mise à jour ne
  la rend pas contractante si L > 1, et un point fixe peut être faux.

## Ce que nous faisons

1. Corriger `balanced_perm` (pas premier avec m) pour les prochaines campagnes QCM.
2. Expérience 3 : **l'appel du minoritaire**, sur ARC dev puis test.
3. Ajouter au coordinateur de l'expérience 2 le **certificat d'arrêt** (2), d'abord en rejeu sur les
   résultats GSM8K.
