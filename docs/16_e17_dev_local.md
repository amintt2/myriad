# E17 — mise au point DEV locale R00

La décision du 13 octobre à 12 h 40 autorise la mise au point locale. Ce lanceur prépare uniquement
les trois DEV historiques : affine-cipher, beer-song, book-store. L'audit parent précède le smoke officiel
et toute inférence. Aucun téléchargement, installation, serveur distant, traqueur ou identité n'est nécessaire.

Le contrôleur et les vérificateurs restent dans le runtime Python WSL existant du pilote Aider, avec
Landlock ABI ≥ 3, namespaces et seccomp, sans repli. Le serveur Windows llama.cpp b11505/ff5888f99 utilise
Vulkan0/RX 6650 XT, 8 Go, un modèle et un slot à la fois, contexte 32 768, réflexion 0, Jinja activé.
Il écoute seulement sur le loopback Windows. WSL NAT ne peut pas joindre ce loopback directement : un
processus Python Windows de confiance relaie les requêtes par stdin/stdout, puis utilise le client HTTP
et le client séquentiel existants. Aucune modification réseau, WSL ou pare-feu n'est requise.

Le job Windows du relais contient ses enfants et les tue à sa fermeture, y compris après EOF du contrôleur.
Le lanceur n'adopte aucun PID ni serveur existant. Le test synthétique Windows vérifie aussi qu'un processus
étranger reste vivant. Chaque chargement a son propre journal. Une fermeture non confirmée est une erreur.
Le lecteur stdin utilise le pipe Windows sans verrou du tampon Python ; il est arrêté et joint avant toute
finalisation normale, preflight/smoke compris. L'accusé `closed` exige ensuite un vrai retour du worker de 0.
L'EOF pendant une requête est une interruption anormale (143), jamais une preuve de fin normale.

Chaque requête et réponse porte une identité de session et un compteur strict. Un délai de réception
conserve les octets partiels et l'identité en attente. Le contrôleur draine uniquement cette réponse,
sous borne de 15 s dans les 7 200 s de session, avant toute requête suivante. La réponse drainée ne fournit
aucune réponse à un nouvel épisode ; une identité incohérente ou une panne d'infrastructure reste fatale.
Le drainage avant un nouvel appel consomme son budget, et celui de fin d'épisode clôt les compteurs
avant de commencer la tâche suivante. Aucun épisode ni appel modèle n'est rejoué.

Un timeout HTTP renvoie une issue typée de génération après arrêt de notre serveur. Le harnais R00
l'interprète comme sa condition normale de délai, puis note les sources courantes. Le worker reste vivant
et recharge seulement le modèle demandé par le prochain appel. Les autres erreurs ne deviennent pas des
délais de génération. Le client HTTP existant est conservé : son champ Bearer reçoit le libellé public
fixe `local-loopback`, évitant l'en-tête vide invalide ; aucune authentification, clé privée ou identité
n'est créée sur ce serveur loopback sans option d'authentification.

## Ressources et provenance

L'archive officielle Windows Vulkan déjà présente est vérifiée contre les épingles de
`app/myriad/llamacpp.py` : 33 442 611 octets, SHA-256
`ace19f118b9e32382ec9b9b123aa19e8fdf54981fb82e82977b5298efaf01d49`.
Chaque membre installé est comparé à cette archive, binaire et DLL inclus. Le préflight observe seulement
`--version` et `--list-devices`, sans charger de modèle. Le modèle, sa taille, sa révision GGUF et son hash
reprennent les déclarations canoniques de `phase0/e17_campaign.py`, cohérentes avec le catalogue :

| Modèle Q8_0 | Octets | SHA-256 |
| --- | ---: | --- |
| Qwen3.5-2B | 2 012 012 800 | `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1` |
| Granite4.2-3B | 3 892 651 552 | `fbe986738041418e26de9e123ba740cb654931f85bf572a71bd01f9e6b85e53d` |

`phase0/speeds_consumer.json` contient respectivement 36 et 56 jetons/s, mesurés par llama-bench tg128.
Ce sont des débits historiques de décodage, pas une mesure du nouveau harnais, du prompt, du chargement
ou d'un vote. Les GGUF peuvent tenir un par un ; l'allocation effective du contexte et les délais restent
à observer. Une erreur de chargement ou un manque de mémoire n'entraîne aucune réduction de contexte,
quantification, budget ou bascule CPU.

`runtime.json` enregistre les observations Windows, les hashes des fichiers serveur, Python Windows/WSL,
le lock, les versions installées et les sources utilisées. `provenance.json` réutilise le validateur existant.
Les hypothèses énergétiques restent 100 W et 0,25 EUR/kWh, non mesurées. Les archives, GGUF, runtime et
cache DEV sont uniquement lus. La vue copiée contient uniquement les trois DEV et leurs attributions ;
les métadonnées historiques de sélection restent intactes. Aucun fichier scientifique TEST n'est ouvert.

## Commandes parent après audit

Depuis WSL, définir les ressources existantes suivantes, sans installer ni télécharger quoi que ce soit :

```bash
export MYRIAD_ROOT="$(pwd)"
export MYRIAD_ASSETS="$(wslpath -u "$MYRIAD_ASSETS_WINDOWS")"
export MYRIAD_PILOT_RUNTIME="$HOME/.local/share/myriad-code-pilot/venv"
# MYRIAD_WINDOWS_PYTHON : python.exe existant, chemin WSL ; httpx disponible.
# MYRIAD_DEV_CACHE : cache Aider existant, chemin WSL ; seuls les trois DEV seront lus.
# MYRIAD_LOCAL_PROOFS : dossier parent existant, neuf pour cette campagne.
common=(--runtime "$MYRIAD_PILOT_RUNTIME" --windows-python "$MYRIAD_WINDOWS_PYTHON"
        --binary "$(wslpath -w "$MYRIAD_ASSETS/tools/llama.cpp-bin/llama-server.exe")"
        --archive "$(wslpath -w "$MYRIAD_ASSETS/tools/llama-b11505-bin-win-vulkan-x64.zip")"
        --models-dir "$(wslpath -w "$MYRIAD_ASSETS/models")" --tasks-dir "$MYRIAD_DEV_CACHE")
cd "$MYRIAD_ROOT/phase0"
"$MYRIAD_PILOT_RUNTIME/bin/python" e17_local.py preflight "${common[@]}" \
    --output "$MYRIAD_LOCAL_PROOFS/preflight-single"
"$MYRIAD_PILOT_RUNTIME/bin/python" e17_local.py smoke "${common[@]}" \
    --output "$MYRIAD_LOCAL_PROOFS/smoke-single"
"$MYRIAD_PILOT_RUNTIME/bin/python" e17_local.py run "${common[@]}" --mode single \
    --output "$MYRIAD_LOCAL_PROOFS/run-single"
# Examiner et conserver cette tentative avant la comparaison séquentielle.
"$MYRIAD_PILOT_RUNTIME/bin/python" e17_local.py run "${common[@]}" --mode vote \
    --output "$MYRIAD_LOCAL_PROOFS/run-vote"
```

Depuis Git Bash, préfixer l'appel du script WSL par `MSYS_NO_PATHCONV=1`.
Chaque sortie doit être absente : aucune reprise implicite ni écrasement, même pour preflight/smoke.
Les références officielles et les négatifs passent avant la première requête modèle de chaque tentative.
Les deux mesures sont distinctes : single d'abord, vote Qwen puis Granite ensuite. La boucle
`aider_agent`/`code_pilot.run_one` et les budgets R00 restent identiques : 4 096 jetons par appel,
49 152 réservés par épisode, 20 étapes, 900 s d'agent, 30 s par commande, 120 s de vérification finale,
seed 0. Le chargement est inclus dans le délai de requête ; le vote est une simulation locale séquentielle,
sans démonstration distribuée. Le relais a une borne de session de 7 200 s ; ce plafond n'augmente aucun budget.

## Lecture des tentatives et limites

`registry.json` conserve chaque tentative commencée, les hashes de transcriptions et les temps de chargement
et requêtes par épisode. Les modes non mesurés restent inconnus. `complete=true` exige les trois épisodes
graded et une fermeture du serveur confirmée ; une interruption laisse `complete=false`, éventuellement
`failure.json`, sans résumé ni score. Un candidat peut échouer avec une notation complète. Aucune sortie
partielle n'est une note de campagne. Les traces gardent le comportement R00, y compris refus de soumission
sans test officiel vert et traitement des réponses tronquées.

Le chemin Windows natif et les pipes ont été testés avec des processus et réponses synthétiques. La
connexion réelle au modèle, l'allocation Vulkan à ce contexte, le smoke officiel sur cette combinaison et
la mesure DEV attendent l'audit parent. Cascade, 27B, BigCodeBench, TEST, gel et article restent hors périmètre.
Ne pas présenter un smoke synthétique comme une validation scientifique ni comme une vraie inférence.
