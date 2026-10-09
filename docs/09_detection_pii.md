# Détection locale des données personnelles avant l'envoi (couche « modèle »)

> État : intégré au garde de la passerelle, facultatif et expérimental. Le modèle Wismut MIT est le seul
> modèle du catalogue runtime. **La calibration de cette intégration n'est pas validée** : aucun taux
> de fuite garanti n'est revendiqué. Les tableaux des sections 3 à 5 et du rapport du banc décrivent
> la branche source `pii-detector` (`e254079`, correctif `06eba72`), avant l'unification des règles.
> Ils ne mesurent pas la politique intégrée. [Rapport historique](../app/bench/results/pii_report.md).
> L'audit de la branche source reste archivé dans le dépôt privé.

### État de l'intégration (tâche 5b)

[`privacy_rules.py`](../app/myriad/privacy_rules.py) contient le jeu canonique de détections du garde,
du scanner et du banc. `matches()` conserve les détections non fusionnées et leurs types fins pour les
préférences (`api_key`, `private_key`, `card`, `iban`, `path`, `name`, etc.) ; `find()` adapte ces mêmes
détections aux types larges du banc. [`privacy.py`](../app/myriad/privacy.py) conserve les modes
`off`/`warn`/`mask`/`local`, les termes personnalisés, les marqueurs réversibles, les confirmations et la
fusion conservatrice des recouvrements. Les alias et formats hérités sont conservés, y compris `pwd`,
secrets numériques, points ou parenthèses et noms de comptes génériques dans les chemins. Cela conserve
aussi les alertes prudentes sur des exemples de clés ou des expressions de code affectées à un secret.

Le raccord existant appelle [`privacy_model.scan()`](../app/myriad/privacy_model.py). Il ignore les
passages `source=rules` déjà traités par le garde et utilise `model_decision`, décision additionnelle
du modèle seul. Le module partagé accepte `include_rules=False` pour éviter un second scan des mêmes
règles ; `scan(texte)` conserve son contrat autonome par défaut. La politique autonome (`decision`) ne
force donc pas les règles en local dans la passerelle. Aucun scan ne télécharge de fichier ni n'envoie
le texte à un service distant.

En l'absence normale des fichiers du modèle ou des paquets facultatifs, le garde habituel s'applique.
Des fichiers partiels, de taille incorrecte ou d'empreinte incorrecte sont une corruption, même sans
runtime ; une erreur de chargement ou d'inférence demande confirmation avant toute sortie du texte.
Cette retenue s'applique aussi si les règles ont déjà masqué une partie du message. Une confirmation
explicite autorise l'envoi selon le comportement existant ; un mode `local` reste prioritaire.
Les métadonnées des fichiers sont recontrôlées à chaque scan : une installation terminée ou une réparation
relance la vérification des tailles et SHA-256 puis le chargement, sans redémarrage. Des fichiers inchangés
gardent leur vérification ou leur état de corruption en cache, sans recalcul lourd de leurs empreintes.

L'unification des règles et les préférences de masquage invalident l'ancienne calibration combinée.
Les seuils bas livrés valent donc zéro, `guarantee=False` et `n_cal=0` : quand Wismut fonctionne,
**chaque message non vide demande confirmation**. Les passages affichés du modèle utilisent alors
le seuil descriptif 0,5 ; ils ne prouvent pas une couverture complète. Une calibration de la politique
intégrée reste nécessaire avant de rétablir un envoi automatique fondé sur un seuil non nul.

Le catalogue runtime épingle Wismut à `4348999cd3c2e20c49615e9af7c6bbb45b64cd85`, avec tailles et SHA-256
des trois fichiers. Installation volontaire : extra `pii`, puis `ensure_model()` (voir
[le guide app](../app/README.md#détection-locale-des-données-personnelles-facultative-expérimentale)).
**Rampart est rejeté (CC-BY-4.0)** : son entrée de téléchargement et les parcours exécutables ont été
supprimés. Les mentions et mesures historiques ci-dessous expliquent la comparaison, sans autoriser
son installation ni son exécution.

Les corrections de l'audit couvrent l'adresse complète `12 rue des Lilas, 75011 Paris`, les bornes de
l'IBAN suivi de prose ou d'un autre IBAN, les courriels contenant `%` et les sous-ensembles de runs à passages :
analyse, calibration résiduelle et
replay ne parcourent que les IDs disponibles. Les fixtures du banc et les tests du garde/passerelle
ne font aucun appel réseau externe. Les mesures historiques n'ont pas été recalculées pour cette
intégration ; le comportement prudent ci-dessus tient compte de cette limite.
L'extraction des clés d'affectation parcourt les candidats entiers sans reprises à l'intérieur des longues
chaînes de tirets ou de caractères ; les alias et valeurs hérités restent détectés. Des tests en
sous-processus contrôlent des négatifs de la taille maximale d'une requête, avec un délai généreux.

## 1. Le problème

Myriad envoie la question de l'utilisateur à des pairs. La branche `security` ajoute le chiffrement de
bout en bout et un garde-fou à base d'expressions régulières qui remplace les secrets et les données
personnelles par des marqueurs réversibles. Le propriétaire veut une seconde couche, fondée sur un
modèle local qui donne **une probabilité** que le texte contienne des informations personnelles, et qui
retient l'envoi en cas de doute, parce qu'« on ne sait jamais s'il se trompe ou s'il ne l'a juste pas
détecté ».

Contraintes : environ 150 à 250 Mo de mémoire, processeur seul, environ 50 ms pour une question typique
sur un i5-10400F, français et anglais (et si possible les autres langues de l'UE), et du code (clés
d'API dans les sources, chemins, noms d'utilisateur).

La question centrale n'est pas « quel modèle a le meilleur F1 », mais : **avec quelle garantie peut-on
dire qu'au plus α % des messages qui contiennent des données personnelles partiront quand même ?**

## 2. Les candidats

Relevés sur l'API de Hugging Face le 9 octobre 2026 (révision épinglée, licence de la fiche et du
dépôt, fichiers et tailles). Règle du dépôt : poids ouverts, licence Apache-2.0, MIT ou BSD, pas de
`trust_remote_code`, poids en safetensors ou en format de données (ONNX, GGUF) chargé sans exécuter de
code du dépôt.

### 2.1 Ceux que le propriétaire a cités

| candidat (révision) | licence | format, exécution | langues | taille | verdict |
| --- | --- | --- | --- | --- | --- |
| `convaiinnovations/laya` (7b928d8) | Apache-2.0 | safetensors ; architecture maison `LayaTypedDecisions`, il faut le paquet `laya` (le dépôt contient aussi `rl_agent_api.py`, `rl_common.py`) ; pas d'ONNX | anglais (ModernBERT-large, 421 M) ; `multilingual/` : mmBERT-base, 322 M, 100+ langues | 843 Mo (EN), 644 Mo (multilingue) | **hors budget** (≥ 650 Mo de poids ; la fiche annonce 193 à 464 ms par requête sur CPU) et exécution par du code tiers. Mesuré à travers sa version PII ci-dessous. |
| `impacte/bunker-laya` (f18245a) | Apache-2.0 pour les poids, **mais** entraîné entre autres sur `lmsys/toxic-chat` et `HuggingFaceH4/no_robots` (CC-BY-NC-4.0) et `ai4privacy/pii-masking-400k` (licence propre) : risque de licence | ONNX FP32 de 1,69 Go ; tête de décision maison (`marker_pos`, `marker_mask`, `qtype`) ; l'INT8 n'est pas publié (« la tête PII s'effondre ») ; reconstruit en 40 lignes avec onnxruntime, sans code du dépôt | anglais (ModernBERT-large) | 1,69 Go | **mesuré** (section 5) ; hors budget mémoire |
| « pompom » (getpompom.com) | – | – | – | – | **aucun lien trouvé** : getpompom.com est un logiciel de montage de podcasts pour Mac (Core ML, transcription), sans rapport avec Laya, bunker-laya ou la détection de données personnelles. Le propriétaire pense qu'il utilise Gemini Embedding : c'est une **API en ligne de Google**, donc le texte partirait chez Google, ce qui contredit le but. Exclu ; ses cousins ouverts sont testés plus bas. |
| `knowledgator/gliner-pii-small-v1.0` (d21aad5) | Apache-2.0 | ONNX (quint8 83 Mo, fp16 164 Mo, fp32 327 Mo) ; GLiNER à jetons (squelette ettin-encoder-68m) ; contrat d'entrée reconstruit (pas besoin du paquet `gliner`, ni de son `pytorch_model.bin` en pickle) | anglais (étiquette « multilingual ») | 83 Mo | **mesuré** |
| `kalyan-ks/ettin-17m-nemotron-pii`, `-68m` (3b4efba, 500262a) | MIT | safetensors seulement (34 / 137 Mo), `ModernBertForTokenClassification` (natif dans transformers, pas de code distant) ; il faudrait PyTorch ou un export ONNX maison | anglais | 34 / 137 Mo | non mesurés tels quels : leur export ONNX publié par `rulesentry-io` (32m, 128 Mo, MIT) est **mesuré** comme représentant de la famille |
| Rampart (`nationaldesignstudio/rampart`, b1993e4), utilisé par Arcjet | **CC-BY-4.0**, hors de la règle Apache/MIT/BSD | ONNX 4 bits 14,7 Mo (MiniLM-L6, 18,5 M) | EN, ES, FR, DE, IT, PT, NL | 14,7 Mo | **rejeté**, non téléchargeable et non exécutable par ce banc ; mesures historiques seulement, optimistes sur openpii utilisé à l'entraînement |
| OpenAI Privacy Filter (`openai/privacy-filter`, 7ffa9a0) | Apache-2.0 | architecture `openai_privacy_filter` ; plus petit ONNX : q4, 917 Mo | multilingue | ≥ 0,9 Go | **hors budget**, non mesuré |
| spaCy « à la Presidio » | `en_core_web_sm` MIT ; `fr_core_news_sm` **LGPL-LR** | pipeline spaCy | EN ; FR exclu par la licence | ~12 Mo + spaCy | non mesuré : NER générique (personnes, lieux, organisations) ; la couche de règles de Myriad joue déjà le rôle des « recognizers » de Presidio, et le modèle français n'est pas sous licence permissive |

### 2.2 Ceux trouvés en cherchant

| candidat (révision) | licence | format | langues | taille | verdict |
| --- | --- | --- | --- | --- | --- |
| `Wismut/nym-pii-multilingual-small` (4348999) | MIT ; données d'entraînement MIT (synthétiques) + passages de Wikipédia étiquetés par un LLM (non redistribués) | ONNX `edge-int8` (108 Mo) et `int8` (139 Mo), ModernBERT (mmBERT-small élagué à 16 couches) | 22 langues dont FR, EN, DE, ES, IT, PT, NL, PL, SV, CS | 108 Mo | **mesuré, retenu** |
| `onnx-community/bert-small-pii-detection-ONNX` (6cb4e77) | Apache-2.0 | ONNX int8 | anglais | 29 Mo | mesuré |
| `Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection` (e43200a) | MIT | ONNX fp32 | anglais (Nemotron-PII) | 90 Mo | mesuré |
| `ai4privacy/llama-ai4privacy-*-openpii`, `onnx-community/multilang-pii-ner-ONNX` | étiquetés MIT, **mais** entraînés sur `open-pii-masking-500k`, soumis à la licence communautaire Llama 3.1/3.3 (nom « Llama… », mention « Built with Llama ») | ONNX | multilingue | 151 à 279 Mo | **exclus** (licence héritée) |
| `iiiorg/piiranha-v1` | CC-BY-NC-ND-4.0 | – | 6 langues | 0,7 Go | **exclu** |
| `bardsai/eu-pii-anonimization-multilang` | Apache-2.0 | ONNX int8 279 Mo (XLM-R base) | 24 langues de l'UE | 279 Mo | hors budget, non mesuré |
| famille `OpenMed-PII-*` | Apache-2.0 | safetensors seulement, ≥ 44 M (DeBERTa) | EN, ou une langue par modèle | ≥ 170 Mo | non mesurés (pas d'ONNX, langues séparées) |

### 2.3 Plongements de phrases + sonde (piste « Gemini Embedding »)

Un modèle de plongements donne un vecteur par texte ; une petite régression logistique (ou un MLP à
une couche cachée de 128) entraînée sur `train` donne p(contient des données personnelles).
**Une sonde au niveau du document ne donne aucun passage** : elle ne peut pas masquer, seulement
décider envoyer / demander / garder en local. Elle est donc évaluée comme porte, combinée aux règles et
à un modèle à passages.

| candidat (révision) | licence | format | langues | taille |
| --- | --- | --- | --- | --- |
| `google/embeddinggemma-300m` | **Gemma Terms of Use** (non permissive), accès restreint | – | 100+ | **exclu** |
| `google/embeddinggemma-2` (Apache-2.0, génération Gemma 4) via `onnx-community/embeddinggemma-2-ONNX` (daa72c5) | Apache-2.0 | ONNX q4, encodeur texte seul (174 Mo + 0,5 Mo) | 100+ | 175 Mo |
| `intfloat/multilingual-e5-small` (614241f) | MIT | ONNX qint8 | ~100 | 118 Mo |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (e8f8c21) | Apache-2.0 | ONNX quint8 | 50+ | 118 Mo |
| `thenlper/gte-small` (17e1f34) | MIT | ONNX qint8 | anglais | 34 Mo |
| `BAAI/bge-small-en-v1.5` | MIT | ONNX fp32 133 Mo | anglais | non mesuré (proche de gte-small) |
| `jinaai/jina-embeddings-v2-small-en` | Apache-2.0 | demande `trust_remote_code` (JinaBert) | anglais | **exclu** |

## 3. Le banc d'essai

`app/bench/pii/` (non livré avec l'app ; dépendances dans le groupe `pii-bench`) :

```
uv sync --group pii-bench
uv run --group pii-bench python -m bench.pii.fetch                # modèles et données, révisions épinglées
uv run --group pii-bench python -m bench.pii.run_bench            # un processus neuf par détecteur
uv run --group pii-bench python -m bench.pii.run_bench --mem-all  # mémoire et latence sans les données
uv run --group pii-bench python -m bench.pii.analyze              # bench/results/pii_report.{json,md}
```

### 3.1 Les données

| source (révision) | licence | contenu | utilisé |
| --- | --- | --- | --- |
| `nvidia/Nemotron-PII`, test (b70ffaf) | CC-BY-4.0 | documents synthétiques anglais, 55 types, passages au caractère près | 1 200 tirés au hasard (tous positifs) |
| `ai4privacy/pii-masking-openpii-1.5m`, validation (a785eb5) | CC-BY-4.0 (champ « other » du dépôt, licence CC-BY-4.0 dans la fiche) | phrases à gabarits, 30 langues | 30 plages d'octets réparties sur le fichier de 1 Go (trié par langue) ; 300 EN, 300 FR, 120 DE/ES/IT/NL/PT, 30 par autre langue |
| `gretelai/synthetic_pii_finance_multilingual`, test (7b844d1) | Apache-2.0 | documents financiers synthétiques, 7 langues | 300 par langue ; **hors calibration** (étiquettes automatiques bruitées hors anglais : « Damen », « Lizenznehmer », « [E-Mail] » étiquetés comme noms ou courriels) |
| `OpenAssistant/oasst1`, train (fdf72ae) | Apache-2.0 | vraies premières questions d'utilisateurs | 500 EN, 247 FR, 40 par autre langue ; **négatifs présumés** (non étiquetés : ils citent parfois des personnes publiques) |
| `openai/openai_humaneval` (7dce605) | MIT | 164 énoncés Python | négatifs « code » |
| jeu fait main (`bench/pii/handmade.py`) | – | questions en FR/EN et code : courriels, téléphones FR et internationaux, IBAN et cartes valides (Luhn), NIR avec clé, adresses, noms, clés AWS/GitHub/OpenAI/Stripe/HF, clé privée, JWT, chemins avec nom d'utilisateur, `ssh user@ip` ; négatifs difficiles : personnes publiques, code sans secret, `os.environ[...]`, `YOUR_API_KEY`, clé d'exemple AWS, empreintes, UUID, numéros qui échouent au Luhn, versions `1.2.3.4` | 160 positifs, 58 négatifs |

Le jeu REDACT (25 langues) n'est pas utilisé : son miroir Hugging Face demande d'accepter des
conditions propres (« REDACT Dataset Terms ») et `pii-masking-400k` d'ai4privacy a une licence
maison : ni l'un ni l'autre n'est sous licence permissive claire.

Les étiquettes sont ramenées à dix types : PERSON, EMAIL, PHONE, ADDRESS, ID, FINANCIAL, SECRET, IP,
USERNAME, DATE_OF_BIRTH. Ce qui n'identifie pas seul une personne (ville, pays, entreprise, date, âge,
genre, métier, URL, codes bancaires) est ignoré. Un document est **positif** s'il contient au moins un
passage de ces types.

### 3.2 Le protocole

* Découpage fixe par empreinte de l'identifiant : `train` 30 % (entraînement des sondes, choix de λ_h),
  `cal` 30 % (seuil conforme), `test` 40 % (tous les chiffres rapportés). `dev` = `train` + `cal`.
* Chaque détecteur tourne dans un processus neuf. Les modèles à passages passent par le code de l'app
  (`OnnxTokenClassifier`) : fenêtres de 512 jetons qui se chevauchent de 64, jamais de troncature ; le
  score d'un jeton est 1 − P(O) − P(étiquettes hors périmètre).
* Mémoire : RSS et pic de l'ensemble de travail **mesurés par le système** (psutil, `peak_wset` sous
  Windows), dans un processus qui ne charge que l'exécution et le modèle (`--mem`), jamais lus sur les
  fiches. Latence : un document à la fois, onnxruntime sur les 6 cœurs du i5-10400F.

## 4. La garantie statistique

### 4.1 Le score critique d'un document

Le détecteur donne à chaque jeton t d'un texte x un score s_t(x) ∈ [0, 1] : la masse de probabilité des
étiquettes « donnée personnelle » (1 − P(O) − P(étiquettes hors périmètre)). Une règle qui se déclenche
(courriel, IBAN valide, clé d'API…) vaut 1 sur ses caractères. Pour un seuil λ :

* politique **« bloquer »** : le document est retenu si max_t s_t(x) ≥ λ (ou si une règle se
  déclenche). Son score critique est c(x) = max_t s_t(x) ;
* politique **« masquer »** (celle retenue, § 5.3) : les passages des règles sont masqués ; le modèle ne
  juge que les jetons R(x) qui ne touchent aucun passage des règles, et le document est retenu si
  max_{t ∈ R(x)} s_t(x) ≥ λ. Il fuit s'il contient un passage personnel que les règles ne masquent pas
  **entièrement** et qu'il n'est pas retenu. Son score critique est c(x) = max_{t ∈ R(x)} s_t(x) si un
  passage n'est pas entièrement masqué par les règles, +∞ sinon.

Dans les deux cas, un document qui contient des données personnelles est **raté** au seuil λ si et
seulement si c(x) < λ. La perte L(λ) = 1{c(x) < λ} est croissante en λ : baisser le seuil ne peut que
réduire les ratés (au prix de fausses alertes).

On a aussi mesuré une troisième variante, « le modèle masque lui-même tous les jetons ≥ λ », avec le
critère c(x) = min_g max_{t ∩ g ≠ ∅} s_t(x) (chaque passage g est *touché* par un jeton ≥ λ). Ce
critère est optimiste (un passage touché n'est pas forcément masqué en entier) et pourtant il ne se
calibre à aucun niveau utile (§ 5.3) : cette variante est écartée.

### 4.2 Le seuil conforme (garantie marginale)

On dispose de n documents de calibration qui contiennent des données personnelles, de scores critiques
c_1, …, c_n, et d'un nouveau document positif de score c_{n+1}. **Hypothèse : les n + 1 scores sont
échangeables** (par exemple tirés indépendamment de la même loi ; le détecteur est fixé avant de voir la
calibration).

Soit k = ⌊α (n + 1)⌋ et λ̂ = c_(k), la k-ième plus petite valeur parmi c_1, …, c_n (λ̂ = 0 si k = 0,
c'est-à-dire « tout retenir »). Alors

  P(c_{n+1} < λ̂) ≤ k / (n + 1) ≤ α.

*Preuve.* Sans ex æquo, le rang R de c_{n+1} parmi les n + 1 valeurs est uniforme sur {1, …, n + 1} par
échangeabilité, et il y a exactement R − 1 valeurs de calibration sous c_{n+1}. Si R ≤ k, il y en a au
plus k − 1 en dessous, donc c_(k) > c_{n+1}. Si R > k, il y en a au moins k en dessous, donc
c_(k) < c_{n+1}. L'événement « raté » est donc exactement {R ≤ k}, de probabilité k / (n + 1). Avec des
ex æquo, on départage au hasard : l'événement strict c_{n+1} < c_(k) implique toujours R ≤ k, la
probabilité ne peut que baisser. (L'inégalité doit rester stricte : avec « ≤ », des scores tous égaux
donneraient un raté sûr.) ∎

C'est le contrôle conforme du risque (Angelopoulos et al., 2022) pour une perte 0/1 croissante, qui se
réduit ici au quantile conforme classique. Conséquences pratiques :

* il faut au moins n ≥ 1/α − 1 positifs de calibration pour que k ≥ 1 : **99 pour α = 1 %**. En dessous,
  la seule règle sûre est de tout retenir. Même au-delà, le seuil peut valoir 0 (des positifs sans aucun
  jeton signalé) ; l'app traite alors **tout texte non vide** comme sensible (« demander » ou « local »),
  y compris celui sur lequel le modèle ne signale aucun jeton ;
* la garantie est **marginale** : elle porte sur la moyenne sur les tirages de la calibration et du
  nouveau document, et elle est conditionnelle à « le document contient des données personnelles »
  (taux de faux négatifs), pas au type de donnée ni à la langue ;
* l'hypothèse porte sur les **scores des positifs**, le détecteur étant fixé : entraîner les sondes sur
  `train` (disjoint de `cal` et `test`) et y choisir λ_h ne la casse pas. Exclure gretel ne la casse pas
  non plus, mais restreint la population couverte : rien n'est garanti pour gretel, ni pour un mélange de
  sources aux proportions différentes.

### 4.3 Variante « avec forte probabilité » (PAC)

On peut vouloir que, pour la calibration effectivement tirée, le taux de ratés du seuil choisi soit
≤ α avec probabilité ≥ 1 − δ. Il faut ici une hypothèse **plus forte** que l'échangeabilité : les scores
de calibration des positifs sont **indépendants et de même loi** F (le détecteur étant fixé). Par
couplage C_i = F⁻¹(U_i) avec des U_i uniformes indépendantes, le taux de ratés du seuil c_(k) vérifie
F(c_(k)−) ≤ U_(k), où U_(k) suit une loi Bêta(k, n − k + 1) (statistique d'ordre de n uniformes ; la
limite à gauche rend l'inégalité valable aussi pour des scores discrets). Donc

  P(taux de ratés > α) ≤ P(Bêta(k, n − k + 1) > α) = P(Bin(n, α) ≤ k − 1),

et l'on prend le plus grand k tel que P(Bin(n, α) ≤ k − 1) ≤ δ (cette probabilité croît avec k). Pour
α = 1 % et δ = 5 %, il faut au moins n = ⌈log δ / log(1 − α)⌉ = 299 positifs pour que k ≥ 1, ce qui
rend un seuil non nul *possible* (des scores nuls peuvent encore l'imposer à 0). Avec n = 944, k = 5.
`bench/pii/conformal.py` calcule les deux seuils (Vovk, 2012, § 3).

### 4.4 Ce que l'on vérifie sur le test

Le seuil est choisi sur `cal` seulement. Sur `test` (jamais vu), on compte les positifs ratés m sur
N et l'on donne l'intervalle de Clopper-Pearson à 95 % de m / N (couverture au moins nominale sous un
modèle binomial). Ce n'est **pas** une preuve de la garantie (elle est démontrée sous l'hypothèse) mais
un contrôle. L'intervalle estime le risque *du seuil tiré*, alors que la garantie marginale est une
moyenne sur les calibrations : un intervalle au-dessus de α peut venir d'une calibration défavorable
sans aucune rupture d'échangeabilité ; un écart répété signalerait une rupture.

On re-tire aussi 500 fois la partition `cal`/`test` des positifs (mêmes tailles) : la moyenne des taux
de ratés doit rester proche de α ou en dessous. Ce contrôle vérifie le mécanisme sur ce corpus, pas sa
représentativité, et une garantie marginale ne dit rien de la *part* des tirages qui dépassent α.

### 4.5 La zone grise et la règle de décision

Le seuil bas λ̂ est fait pour ne rien laisser passer ; il signale aussi beaucoup de textes anodins. On
ajoute un seuil haut λ_h, choisi sur `train` (juste au-dessus du score qui laisse 5 % des négatifs de
`train` au-delà). Règle appliquée par `privacy_model.decide` :

| situation | « bloquer » | « masquer » |
| --- | --- | --- |
| aucune règle, aucun passage du modèle ≥ λ̂ | **envoyer** | **envoyer** |
| règles seulement (pas de passage du modèle hors des leurs) | **local** | **masquer** les passages des règles, puis envoyer |
| un passage du modèle (hors règles pour « masquer ») entre λ̂ et λ_h | **demander** | **demander** |
| passages du modèle tous ≥ λ_h | **local** | **local** (le modèle ne masque jamais seul) |
| λ̂ = 0 (pas de seuil utile) | **demander** (local si une règle) | **demander** |

« Demander » et « local » retiennent tous deux la question : la zone grise ne change pas la garantie,
qui porte sur l'**envoi automatique**. Un utilisateur qui autorise ensuite l'envoi sort de la garantie.
Le seuil haut s'applique au maximum d'un passage fusionné : un jeton faible collé à un fort peut faire
passer de « demander » à « local », sans fuite supplémentaire.

### 4.6 Ce que la garantie ne dit pas (honnêtement)

1. **L'échangeabilité est l'hypothèse qui porte tout, et elle est fausse en pratique.** La calibration
   mélange des documents synthétiques (Nemotron-PII, ai4privacy) et un petit jeu fait main. Les vraies
   questions des utilisateurs de Myriad sont plus courtes, plus familières, mélangent les langues et
   collent du code. Le taux mesuré sur `test` vaut pour ce mélange, pas pour le trafic réel. Le banc
   montre d'ailleurs des écarts nets entre sources et entre langues (section 5) : la garantie « 1 % »
   vaut **sous les hypothèses d'échantillonnage et de fixation du détecteur, pour la population
   représentée par la calibration**, pas pour chaque sous-groupe (français, code…).
2. **Le choix du détecteur a été fait après avoir vu `cal` et `test`.** Comparer douze détecteurs puis
   garder le meilleur peut rendre la garantie nominale du retenu trop optimiste (sélection). Le choix
   s'appuie surtout sur la licence, la taille, les langues et le taux de faux positifs (une propriété des
   négatifs), mais pour une garantie propre il faut figer le détecteur et ses seuils maintenant, puis les
   recalibrer sur des données neuves.
3. **Indépendance des exemples.** Les variantes d'un même enregistrement (rendus d'un même `uid`
   Nemotron, rendus d'un même gabarit fait main) sont gardées dans la même partition ; mais openpii est
   fait de gabarits, et deux phrases d'un même gabarit peuvent tomber dans `cal` et `test`.
4. La garantie PAC ne vaut ni pour chaque sous-groupe, ni pour chaque lot futur, ni pour l'absence de
   fuite sur toute une utilisation (sur 1 000 questions sensibles, environ 10 peuvent passer).
5. `p_sensitive` est le meilleur score de jeton : un **score** à comparer aux seuils, pas une probabilité
   calibrée que le document contienne des données personnelles. Le mode dégradé (règles seules, sans
   modèle) n'a **aucune** garantie.
6. **Les étiquettes sont supposées justes.** Un raté « mesuré » peut être une erreur d'étiquette, et un
   vrai raté peut être caché par une étiquette manquante. Avec α = 1 %, quelques erreurs d'étiquetage
   suffisent à déplacer le seuil : c'est pourquoi gretel (étiquettes automatiques, bruitées hors anglais)
   est exclu de la calibration et rapporté à part.
7. **Le périmètre est celui des types en jeu** (noms, courriels, téléphones, adresses, identifiants,
   données bancaires, secrets, IP, noms d'utilisateur, dates de naissance). Les catégories particulières
   du RGPD (santé, religion, opinions…) et les combinaisons qui ré-identifient (code postal + maladie
   rare) ne sont pas couvertes.
8. **Ce n'est pas une frontière de sécurité contre un adversaire** qui chercherait à faire passer une
   donnée (encodage, fautes volontaires) : c'est une protection contre les fuites involontaires.
9. Pour rendre la garantie plus proche du réel, il faudra recalibrer sur des questions réelles
   étiquetées (avec le consentement des utilisateurs, localement), et surveiller la dérive. La
   conformité pondérée (Tibshirani et al., 2019) traite un décalage de covariables, mais sous des
   hypothèses précises (même loi des étiquettes sachant le texte, poids connus) : estimer un rapport de
   densités ne rétablit pas automatiquement une garantie exacte.
10. Les seuils du prototype source étaient ceux du banc, diminués de 10⁻⁴ (scores arrondis à 4 décimales dans le banc,
   erreur ≤ 5 × 10⁻⁵). Cela suppose la même exécution (onnxruntime, prétraitement) : tout changement du
   modèle, de l'exécution ou des règles impose de recalibrer.

## 5. Résultats historiques de la branche source

Tous les chiffres : `app/bench/results/pii_report.md` (et `.json`), α = 1 %, δ = 5 %. `cal` contient
944 documents positifs (au moins 99 requis pour α = 1 %, 299 pour la variante PAC) ; `test` 1 266
positifs et 643 négatifs (fait main, oasst, HumanEval, et les rares documents sans passage en
périmètre de Nemotron et openpii), dont 514 questions oasst aussi comptées à part.

**Lacunes connues (à compléter)** : `nym-small-int8` n'a pas été relancé après la correction du lot
(§ 5.6) et n'apparaît plus dans le rapport ; la mémoire et la latence « courte » de nym dans le tableau
du § 5.6 ont été mesurées **avant** les réglages mémoire finaux (voir la note) ; la latence n'a pas été
mesurée avec un llama-server actif sur la même machine.

### 5.1 Ce qui est retenu

**Choix du prototype source : règles + `nym-pii-multilingual-small` (edge int8), politique « bloquer », seuil PAC.**

| | valeur |
| --- | --- |
| seuil bas λ (PAC, δ = 5 %) | 0,0205 (0,0204 dans l'app) |
| seuil haut λ_h (5 % de faux positifs sur `train`) | 0,988 |
| documents positifs envoyés, `test` | **4 / 1 266 = 0,32 %** (IC 95 % : 0,09 – 0,81 %) |
| idem au seuil marginal λ = 0,445 | 10 / 1 266 = 0,79 % (0,38 – 1,45 %) ; moyenne sur 500 re-tirages cal/test : 0,96 % |
| négatifs du banc retenus (PAC) | 20,1 % (100 « demander », 29 « local » sur 643) ; 10,3 % au seuil marginal |
| vraies questions oasst (PAC) | 82,1 % envoyées, 15,2 % « demander », 2,7 % gardées en local (marginal : 92,6 / 4,7 / 2,7 %) |
| français / anglais (seuil marginal) | ratés 2,1 % (3 / 142, IC 0,4 – 6,0 %) / 0,5 % (3 / 592) ; FPR 13,7 % / 7,7 % |
| mémoire (réglages livrés, mesure séparée) | ~58 Mo d'exécution, ~223 Mo modèle chargé, ~231 Mo après inférence ; pic transitoire au chargement (analyse du tokenizer.json de 12 Mo) ≈ 340 Mo estimé, 463 Mo mesuré avant le changement d'ordre de chargement |
| latence (6 cœurs) | questions faites main : p50 12,7 ms, p95 29,9 ms (2 cœurs : 18 / 42 ms) ; < 300 caractères p50 14 ms ; 600 – 1 500 caractères p50 ≈ 116 ms ; > 3 000 caractères p50 ≈ 0,5 s |

Pourquoi lui : c'est le seul candidat sous licence permissive qui tient (à peu près) dans le budget,
couvre le français et les autres langues de l'UE, et garde un taux de faux positifs bas au seuil qui
garantit 1 % de ratés (10,3 % au seuil marginal ; le suivant sous licence permissive, la sonde
EmbeddingGemma 2, n'a pas de passages et a vu les mêmes sources à l'entraînement). Les règles + Rampart
ratent un peu moins mais retiennent deux fois plus de questions anodines (20 % au seuil marginal,
18 % des questions oasst) et Rampart est sous CC-BY-4.0 ; les modèles anglais (bert-small, ettin,
MiniLM, GLiNER) retiennent 27 à 64 % des négatifs.

Le seuil marginal (0,445) donne la garantie « en moyenne » : 0,79 % de ratés sur ce tirage ; sur 500
re-tirages de la partition cal/test, la moyenne vaut 0,96 % et 42 % des tirages dépassent 1 %, ce qui
est compatible avec une garantie marginale (elle borne l'espérance, pas la part des tirages au-dessus).
Le seuil PAC (0,0205), bien plus bas car il dépend de la 5e plus petite valeur de `cal`, double les
faux positifs (20 %) et donne « ≤ 1 % avec 95 % de confiance sur la calibration » : c'est celui que
le prototype source embarquait, puisque le propriétaire préfère ne pas envoyer en cas de doute. Les deux sont
instables : regrouper les variantes d'un même enregistrement dans les partitions (correction demandée
par la relecture) a fait passer le seuil PAC de 0,148 à 0,0205.

### 5.2 Tous les détecteurs (politique « bloquer », seuil marginal α = 1 %)

Voir `app/bench/results/pii_report.md`, § 1 (seuil marginal) et § 1 bis (PAC). Extrait, FPR croissant :

| détecteur | λ | ratés test (IC 95 %) | ratés FR / EN | FPR | FPR oasst |
| --- | --- | --- | --- | --- | --- |
| règles + nym-small-edge | 0,445 | 0,79 % (0,38 – 1,45) | 2,1 / 0,5 % | 10,3 % | 7,4 % |
| sonde EmbeddingGemma 2 (lr) | 0,237 | 0,71 % (0,33 – 1,35) | 0,7 / 0,0 % | 11,8 % | 6,6 % |
| sonde e5-small (lr) | 0,339 | 1,26 % (0,72 – 2,04) | 3,5 / 0,0 % | 12,3 % | 7,6 % |
| règles + Rampart | 0,160 | 0,47 % (0,17 – 1,03) | 0,0 / 0,7 % | 20,2 % | 17,9 % |
| règles + bunker-laya | 0,016 | 0,63 % (0,27 – 1,24) | 0,7 / 0,7 % | 22,6 % | 19,6 % |
| règles + GLiNER-PII small | 0,503 | 0,87 % (0,43 – 1,55) | 0,0 / 0,2 % | 27,4 % | 25,5 % |
| règles + ettin-32m | 0,415 | 1,34 % (0,78 – 2,14) | 1,4 / 0,3 % | 29,1 % | 28,0 % |
| règles + MiniLM-Nemotron | 0,205 | 1,26 % (0,72 – 2,04) | 0,7 / 0,0 % | 43,7 % | 46,3 % |
| règles + bert-small-pii | 0,016 | 0,55 % (0,22 – 1,14) | 0,0 / 0,2 % | 57,2 % | 59,1 % |
| règles seules | 0 | 0 % | – | 100 % | 100 % |

Lecture : au seuil qui garantit 1 % de ratés, ce qui distingue les détecteurs n'est pas le rappel (tous
sont autour de 1 %, par construction) mais **le prix en faux positifs**. Les règles seules ne peuvent
pas atteindre 1 % (29 % des positifs n'ont aucune règle qui se déclenche : noms, adresses, identifiants
sans somme de contrôle) : la seule règle sûre serait de tout retenir.

### 5.3 Politique « masquer »

Masquer tous les jetons au-dessus d'un seuil ne peut pas être calibré à un niveau utile : pour que
**chaque** passage personnel soit couvert dans 99 % des documents, il faut descendre le seuil à 0 (tout
masquer) ; à 10 % de fuites, il retient encore 25 % des négatifs. Les ratés viennent des identifiants
aléatoires, des prénoms isolés, des codes PIN à 4 chiffres.

La politique « masquer » retenue est donc : **les règles masquent** (marqueurs réversibles, comme la
branche `security`), et **le modèle sert de porte sur le reste** (jetons hors des passages des règles).
Si le modèle y voit quelque chose (score ≥ λ_masque), la question n'est pas envoyée automatiquement
(« demander » ou « local ») ; le modèle ne masque jamais seul. Pour les règles + nym : λ_masque =
0,0072 (PAC ; 0,0177 au seuil marginal), 0,63 % de fuites sur `test` (IC 0,27 – 1,24 %), 24 % des
négatifs retenus, 78 % des questions oasst envoyées (règles masquées le cas échéant). Les règles ne
masquent entièrement tous les passages que de 7,6 % des documents positifs : le reste repose sur le
modèle.

### 5.4 Plongements + sonde (la piste « Gemini Embedding »)

Sonde = régression logistique (`-lr`) ou MLP (`-mlp`) sur le plongement de chaque morceau de 256
jetons, entraînée sur `train` (étiquette : le morceau touche un passage personnel) ; score du document =
maximum sur ses morceaux. Seuil marginal α = 1 %, `test` :

| plongement (taille, licence) | ratés | FPR | FPR oasst | FR ratés / FPR | négatifs faits main signalés | RSS chargé / pic (Mo) | p50 / p95 court (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EmbeddingGemma 2, q4 (175 Mo, Apache-2.0) | 0,71 % | 11,8 % | 6,6 % | 0,7 % / 15,4 % | 35 % | 205 / 476 | 65 / 140 |
| multilingual-e5-small, int8 (118 Mo, MIT) | 1,26 % | 12,3 % | 7,6 % | 3,5 % / 17,9 % | 39 % | 434 / 451 | 9 / 25 |
| paraphrase-multilingual-MiniLM, int8 (118 Mo, Apache-2.0) | 1,11 % | 22,6 % | 20,0 % | 2,1 % / 29,1 % | 42 % | 436 / 453 | 11 / 26 |
| gte-small, int8 (34 Mo, MIT, anglais) | 0,47 % | 32,5 % | 31,1 % | 0,0 % / 42,7 % | 58 % | 115 / 130 | 10 / 23 |
| *pour comparer : règles + nym (§ 5.1)* | | | | | | | |

Ce qu'il faut en retenir :

* **Une sonde sur EmbeddingGemma 2 fait presque jeu égal avec nym comme porte** (FPR 11,8 % contre
  10,3 % pour règles + nym), mais elle a un avantage injuste : elle est entraînée sur `train`, qui vient des **mêmes
  sources** que `test` (mêmes gabarits openpii, même style Nemotron), alors que nym n'a jamais vu ces
  corpus. Elle apprend en partie « ce style de document contient des données personnelles » : sur
  gretel, qu'elle a vu à l'entraînement, elle ne rate que 0,6 % des positifs (nym : 9 %), et les
  négatifs faits main, dont les gabarits ne sont pas dans `train`, sont signalés à 35 %. Sur de vraies
  questions, il faudrait l'entraîner sur des données qui leur ressemblent.
* **Elle ne donne aucun passage** : pas de masquage possible, seulement envoyer / demander / local.
* Les modèles e5 et MiniLM multilingues occupent **plus de 430 Mo** une fois chargés (leur table de
  250 000 plongements est décompressée), bien au-delà de leur taille sur disque.
* Combinée aux règles et à nym, une sonde ne réduit presque pas les faux positifs (voir le rapport).

**Est-ce que « la piste pompom » consomme moins que Laya ?** Oui, très nettement : la sonde sur
EmbeddingGemma 2 tient dans ~205 Mo (pic de 476 Mo pendant le chargement), contre 1,5 Go chargé et 2,0 Go
après inférence pour bunker-laya, et elle est trois fois plus rapide (65 ms contre 223 ms). Mais elle ne
consomme pas moins que nym (~223 Mo chargé), qui donne en plus les passages et n'a pas besoin d'être
entraîné sur nos données.

### 5.5 Laya / bunker-laya

bunker-laya a été mesuré sur les 4 797 documents hors gretel (1,1 s par document en moyenne), avec
la question `pii_present` et le contrat de son graphe ONNX (température 4,43 sur la tête « oui/non ») :

| | bunker-laya | règles + bunker-laya |
| --- | --- | --- |
| seuil marginal (α = 1 %) | 0,016 | 0,016 |
| ratés `test` | 0,79 % (IC 0,38 – 1,45 %) | 0,63 % |
| FPR / FPR oasst | 29,4 % / 27,6 % | 22,6 % / 19,6 % |
| FR ratés / FPR | 0,7 % / 34,2 % | 0,7 % / 24,8 % |
| EN ratés / FPR | 1,2 % / 11,3 % | 0,7 % / 8,1 % |
| AUROC | 0,979 | 0,982 |
| RSS chargé / après inférence | 1 501 / 1 984 Mo | |
| latence p50 / p95, questions courtes | 223 / 430 ms | |

Ses probabilités sont très tassées : un texte anodin reçoit environ 0,015 et le seuil qui garantit 1 %
tombe à 0,016. « Calibrée » au sens de sa fiche (score de Brier sur ses propres données de validation)
ne veut pas dire calibrée sur nos questions : au seuil naturel de 0,5 il rate 17 % des documents
anglais et 8 % des français. Avec dix fois la mémoire de nym, il fait moins bien que lui en faux
positifs, surtout en français, et il ne donne pas de passages. Il est écarté.

### 5.6 Coût

Tableau complet : `pii_report.md`, § 7 (processus neuf, RSS mesurée par le système, questions
faites main puis un texte long ; ces mesures « mem » utilisaient encore l'ancien réglage : arène
mémoire d'onnxruntime active, lots de 8 fenêtres, session chargée avant le tokenizer).

| détecteur | RSS chargé / pic (Mo) | p50 / p95 court (ms) |
| --- | --- | --- |
| règles | 58 / 59 | 0,2 / 0,3 |
| Rampart (14,7 Mo) | 85 / 186 | 5,7 / 11,8 |
| nym-small-edge (108 Mo) | 223 / 463 | 12,7 / 29,9 |
| bert-small-pii | 100 / 183 | 4,2 / 10,9 |
| ettin-32m | 201 / 275 | 9,3 / 23,7 |
| GLiNER-PII small (quint8) | 168 / 362 | 61 / 172 |
| sonde EmbeddingGemma 2 (q4) | 205 / 476 | 65 / 140 |
| sonde multilingual-e5-small | 434 / 451 | 9,1 / 24,8 |
| bunker-laya (FP32) | 1 501 / 1 984 | 223 / 430 |

**Note sur nym** (mesures séparées, `OnnxTokenClassifier` actuel) : le pic de 463 Mo vient du
chargement, pas de l'inférence : analyser son tokenizer.json de 12 Mo fait un pic transitoire de
~285 Mo, la création de la session ~190 Mo. L'app charge désormais le tokenizer **avant** la session
(pic attendu ≈ 340 Mo, à re-mesurer de bout en bout), désactive l'arène mémoire et passe une fenêtre
par appel : ~231 Mo après inférence au lieu de ~520 Mo, textes longs ~1,5 fois plus lents.

**Correction en cours de route** : passer plusieurs fenêtres de 512 jetons dans un même appel ONNX
change les sorties des exports ModernBERT (nym, ettin ; écarts de logits jusqu'à 3,4, même sur des
lignes sans remplissage), pas celles des exports BERT (Rampart, bert-small). nym-small-edge et ettin-32m
ont été relancés avec une fenêtre par appel ; les seuils de nym n'ont pas bougé (les documents qui les
fixent tiennent dans une fenêtre). L'app n'utilise plus que des appels à une fenêtre.

### 5.7 Taille de l'installation

L'extra `pii` ajoute onnxruntime 1.31 (roue Windows de 14,9 Mo, 43 Mo installés), numpy (12,6 Mo,
43 Mo installés), tokenizers (2,9 Mo, 8 Mo installés), protobuf et flatbuffers (< 2 Mo) : **environ
31 Mo à télécharger, 100 Mo installés**, plus le modèle (108 Mo de poids + 12 Mo de tokenizer),
téléchargé à la demande dans le dossier de données (`privacy/`). L'installation par défaut et
l'installateur PyInstaller (qui n'installe que l'extra `desktop`) ne changent pas. Si la détection est
un jour activée par défaut, l'installateur grossira d'environ 30 à 40 Mo compressés (onnxruntime et
numpy), sans le modèle.

## 6. Limites et suite

* La garantie est celle du **banc** (§ 4.6) : données majoritairement synthétiques, un petit jeu fait
  main, des négatifs présumés (oasst). Il faut une calibration sur de vraies questions (étiquetées
  localement, avec consentement) avant d'afficher « ≤ 1 % » aux utilisateurs.
* Les personnes publiques (« Victor Hugo », « Simone Veil ») sont signalées : c'est l'essentiel des
  faux positifs en français. La zone grise les transforme en « demander », pas en blocage.
* Les identifiants aléatoires sans contexte (« EIF3Y7L1WC ») et les prénoms isolés restent les ratés
  typiques.
* Le texte est découpé en fenêtres de 512 jetons ; une très longue question coûte plusieurs passages
  du modèle (≈ 0,5 s au-delà de 3 000 caractères) : au-dessus du budget de 50 ms, qui n'est tenu que
  pour les questions courtes.
* L'intégration utilise désormais un seul jeu canonique de règles et le raccord existant du garde
  avant envoi dans la passerelle. La recalibration de cette politique reste à faire ; les anciens
  tableaux ne justifient pas une garantie inchangée après cette modification.

## Références

* A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, *Conformal Risk Control*, 2022
  (arXiv:2208.02814).
* V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, 2005 (quantile conforme
  par découpage).
* V. Vovk, *Conditional validity of inductive conformal predictors*, 2012 (garanties
  « training-conditional », variante PAC).
* R. J. Tibshirani, R. Foygel Barber, E. Candès, A. Ramdas, *Conformal Prediction Under Covariate
  Shift*, 2019.
* C. J. Clopper, E. S. Pearson, *The use of confidence or fiducial limits illustrated in the case of the
  binomial*, Biometrika, 1934.
