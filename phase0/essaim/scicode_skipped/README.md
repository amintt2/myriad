Ces trois fichiers proviennent de `eval/data/` dans le dépôt officiel
[SciCode](https://github.com/scicode-bench/SciCode/tree/e3158ea011d4235245a547460d3688d7ccbf9900/eval/data),
au commit `e3158ea011d4235245a547460d3688d7ccbf9900`. Ils sont conservés sans modification sous licence
Apache-2.0 (copie dans `LICENSE`) ; leurs empreintes SHA-256 figurent dans `essaim/scicode.py` et les manifestes.

Le générateur officiel ignore ces étapes et extrait la définition nommée par leur en-tête pour les étapes
suivantes. Ainsi, `62.1.txt` contient deux classes, mais seule `EnlargedBlock` entre dans la chaîne.
Ces fichiers fournissent du code de contexte, pas un oracle pour les autres étapes du split test.
