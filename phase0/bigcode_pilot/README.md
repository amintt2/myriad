# BigCodeBench DEV pour E17

Notation préparée pour les deux DEV autorisés, **120 et 184**, avec les assertions `unittest` officielles
intactes. Aucun candidat officiel, référence, modèle ou smoke exécuté dans cette livraison.
Le [guide](../../docs/16_e17_bigcode_dev.md) décrit la frontière de confiance, les différences de protocole,
les prérequis WSL, le smoke parent et les commandes de mesure futures.

Port explicite depuis le commit historique `0f7110d` (code `716d3fc`) : lock, runtime, licence, métadonnées
et textes officiels DEV conservés. Le nouveau manifeste conserve l'empreinte du manifeste original.
Les données de 199 restent dans l'archive ancienne ; cette CLI ne les charge jamais. TEST 208/211 reste
réservé, sans contenu, tests ou références dans ces ressources. Aucun téléchargement ni lecture de parquet.
`--prepare --source-cache` copie seulement les deux fichiers DEV nécessaires depuis un cache existant
vérifié vers un dossier neuf. Les références restent dans ce cache privé et ne vont jamais au modèle.

Licence des données et tests : Apache-2.0, BigCode Project / BigCodeBench, Terry Yue Zhuo et collaborateurs.
Les révisions Hard/Full, version v0.1.4 et preuves historiques de sous-ensemble restent dans `tasks.json`.
Python 3.12.15, 31 distributions, binaire, lock, notices et composants compilés sont épinglés dans
`runtime.json`. Licences principales scientifiques BSD ; libgfortran GPL avec exception runtime GCC,
libquadmath LGPL-2.1+ et certifi MPL-2.0 distincts. Aucune wheel redistribuée, aucun environnement
exclusivement permissif revendiqué. Runtime historique réutilisé en lecture seule, sans synchronisation.

`bigcode_execution.py` et `bigcode_rpc.py` conservent le transport historique pour les régressions de sa
limite : enveloppe d'exception forgée et `exit(73)`. Ils ne servent pas à la nouvelle notation.
`adversarial_attestation=false` reste obligatoire : le nouveau vérificateur n'authentifie pas un code
qui attaque volontairement son résultat. Aucun pass@1 officiel, score TEST ou score expérimental revendiqué.
