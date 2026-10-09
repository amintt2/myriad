"""Hand-made French / English / code set: prompts as a user would type them into Myriad.

Templates are written with inline markup `[[TYPE|value]]`; values are drawn with a fixed seed from
small lists, and checksummed identifiers (IBAN, card, NIR) are generated valid. Every value is
synthetic. Hard negatives (label 0) look like PII but are not: public figures, code without secrets,
placeholders, hashes, UUIDs, numbers failing their checksum, documentation example keys."""
from __future__ import annotations

import random
import re
import string

from myriad.privacy_rules import luhn_ok

FIRST = ["Jean", "Marie", "Camille", "Léa", "Hugo", "Nathalie", "Karim", "Fatou", "Thomas", "Inès", "Yanis",
         "Chloé", "Mathieu", "Sophie", "Amina", "Julien", "Élodie", "Olivier", "Mehdi", "Pauline"]
LAST = ["Dupont", "Martin", "Bernard", "Lefèvre", "Moreau", "Benali", "Diallo", "Rousseau", "Garnier", "Nguyen",
        "Lambert", "Fontaine", "Chevalier", "Haddad", "Mercier", "Girard", "Bonnet", "Leroy", "Faure", "Morel"]
FIRST_EN = ["James", "Emily", "Michael", "Sarah", "David", "Olivia", "Daniel", "Grace", "Ryan", "Hannah"]
LAST_EN = ["Smith", "Johnson", "Walker", "Thompson", "Harris", "Clarke", "Turner", "Mitchell", "Cooper", "Hughes"]
STREETS = ["rue de la République", "avenue Jean Jaurès", "boulevard Voltaire", "rue des Lilas", "chemin des Vignes",
           "place du Marché", "impasse des Acacias", "allée des Tilleuls", "quai de la Loire", "rue Victor Hugo"]
CITIES = [("75011", "Paris"), ("69003", "Lyon"), ("13006", "Marseille"), ("31000", "Toulouse"), ("44000", "Nantes"),
          ("67000", "Strasbourg"), ("33000", "Bordeaux"), ("59800", "Lille"), ("35000", "Rennes"), ("06000", "Nice")]
STREETS_EN = ["Baker Street", "Oak Avenue", "Maple Drive", "High Street", "Elm Road", "Kings Road"]
DOMAINS = ["gmail.com", "orange.fr", "free.fr", "laposte.net", "outlook.fr", "proton.me", "wanadoo.fr", "sfr.fr"]
PUBLIC = ["Emmanuel Macron", "Victor Hugo", "Marie Curie", "Albert Einstein", "Napoléon Bonaparte", "Kylian Mbappé",
          "Simone Veil", "Charles de Gaulle", "Ada Lovelace", "Alan Turing", "Linus Torvalds", "Barack Obama"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def card(rng: random.Random) -> str:
    while True:
        d = rng.choice(["4", "51", "52", "53", "54", "55"]) + _digits(rng, 14)
        d = d[:15]
        for c in string.digits:
            if luhn_ok(d + c):
                d += c
                break
        return " ".join(d[i:i + 4] for i in range(0, 16, 4))


def bad_card(rng: random.Random) -> str:
    while True:
        d = "4" + _digits(rng, 15)
        if not luhn_ok(d):
            return " ".join(d[i:i + 4] for i in range(0, 16, 4))


def iban_fr(rng: random.Random) -> str:
    bban = _digits(rng, 10) + "".join(rng.choice(string.digits + string.ascii_uppercase) for _ in range(11)) + _digits(rng, 2)
    n = int("".join(str(int(c, 36)) for c in bban + "FR00"))
    s = f"FR{98 - n % 97:02d}{bban}"
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


def nir(rng: random.Random) -> str:
    body = rng.choice("12") + _digits(rng, 2) + f"{rng.randint(1, 12):02d}" + f"{rng.randint(1, 95):02d}" + _digits(rng, 6)
    key = 97 - int(body) % 97
    return f"{body[0]} {body[1:3]} {body[3:5]} {body[5:7]} {body[7:10]} {body[10:13]} {key:02d}"


def phone_fr(rng: random.Random) -> str:
    d = rng.choice("1234567") + _digits(rng, 8)
    if rng.random() < 0.4:
        return "+33 " + d[0] + " " + " ".join(d[i:i + 2] for i in range(1, 9, 2))
    sep = rng.choice([" ", ".", ""])
    return "0" + d[0] + sep + sep.join(d[i:i + 2] for i in range(1, 9, 2))


def phone_intl(rng: random.Random) -> str:
    return rng.choice(["+44 20 ", "+49 30 ", "+1 415 ", "+32 2 ", "+41 22 "]) + _digits(rng, 3) + " " + _digits(rng, 4)


def rand_token(rng: random.Random, n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def jwt(rng: random.Random) -> str:
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ" + rand_token(rng, 40) + "." + rand_token(rng, 43, string.ascii_letters + string.digits + "-_")


def private_key(rng: random.Random) -> str:
    body = "\n".join(rand_token(rng, 64, string.ascii_letters + string.digits + "+/") for _ in range(4))
    return f"-----BEGIN OPENSSH PRIVATE KEY-----\n{body}\n-----END OPENSSH PRIVATE KEY-----"


POSITIVE = [
    # French
    ("fr", "Bonjour, je m'appelle [[PERSON|{fn} {ln}]] et j'aimerais une lettre de résiliation pour mon forfait."),
    ("fr", "Peux-tu corriger ce mail : « Bonjour Madame, suite à notre échange, vous pouvez me joindre au [[PHONE|{phone}]]. Cordialement, [[PERSON|{fn} {ln}]] »"),
    ("fr", "Mon adresse e-mail est [[EMAIL|{email}]], rédige une réponse polie à mon propriétaire."),
    ("fr", "Rédige un courrier pour changer mon adresse : j'habite maintenant au [[ADDRESS|{num} {street}, {cp} {city}]]."),
    ("fr", "Voici mon RIB pour le remboursement : IBAN [[FINANCIAL|{iban}]], au nom de [[PERSON|{fn} {ln}]]."),
    ("fr", "J'ai payé avec la carte [[FINANCIAL|{card}]] et la banque refuse, que faire ?"),
    ("fr", "Mon numéro de sécu est le [[ID|{nir}]], comment le mettre à jour sur Ameli ?"),
    ("fr", "Résume ce compte rendu : patiente [[PERSON|{fn} {ln}]], née le [[DATE_OF_BIRTH|{dob}]], suivie pour diabète de type 2."),
    ("fr", "Traduis en anglais : « Contactez [[PERSON|{fn} {ln}]] au [[PHONE|{phone}]] ou par mail à [[EMAIL|{email}]]. »"),
    ("fr", "Mon fils [[PERSON|{fn2}]] a de la fièvre depuis hier, il a 4 ans, est-ce que je dois m'inquiéter ?"),
    ("fr", "Écris un message à mon collègue [[PERSON|{fn2} {ln2}]] pour lui dire que la réunion est décalée."),
    ("fr", "J'ai reçu un SMS du [[PHONE|{phone}]] qui me demande de payer une amende, c'est une arnaque ?"),
    ("fr", "Pour la livraison : [[PERSON|{fn} {ln}]], [[ADDRESS|{num} {street}, {cp} {city}]], tél. [[PHONE|{phone}]]."),
    ("fr", "Mon mot de passe Wi-Fi est [[SECRET|{pwd}]], comment le partager avec un QR code ?"),
    ("fr", "Le virement de 1 200 € vers [[FINANCIAL|{iban}]] n'est pas arrivé, rédige une réclamation."),
    ("fr", "Aide-moi à remplir ce formulaire : Nom : [[PERSON|{ln}]] ; Prénom : [[PERSON|{fn}]] ; N° de sécurité sociale : [[ID|{nir}]]."),
    ("fr", "Mon identifiant sur le serveur est dans [[USERNAME|C:\\Users\\{user}]]\\Documents\\factures, comment faire un script de sauvegarde ?"),
    ("fr", "Fais un CV pour [[PERSON|{fn} {ln}]], [[EMAIL|{email}]], [[PHONE|{phone}]], développeur Python depuis 5 ans."),
    ("fr", "Je suis [[PERSON|{fn} {ln}]], né le [[DATE_OF_BIRTH|{dob}]], et je veux contester un PV."),
    ("fr", "Envoie le devis à [[EMAIL|{email}]] et mets en copie [[EMAIL|{email2}]]."),
    # English
    ("en", "Hi, my name is [[PERSON|{fne} {lne}]] and I need help writing a complaint to my bank."),
    ("en", "Please format this: [[PERSON|{fne} {lne}]], [[ADDRESS|{num} {street_en}, London SW1A 1AA]], phone [[PHONE|{phone_intl}]]."),
    ("en", "My card [[FINANCIAL|{card}]] was charged twice, draft an email to support."),
    ("en", "Can you check why my login fails? user [[EMAIL|{email}]] password [[SECRET|{pwd}]]"),
    ("en", "Write a reference letter for my employee [[PERSON|{fne} {lne}]] (reach me at [[EMAIL|{email}]])."),
    ("en", "Our customer [[PERSON|{fne} {lne}]] called from [[PHONE|{phone_intl}]] about an unpaid invoice."),
    # code
    ("code", "Why does this fail?\n```python\nimport boto3\ns3 = boto3.client('s3', aws_access_key_id='[[SECRET|{aws}]]', aws_secret_access_key='[[SECRET|{aws_secret}]]')\n```"),
    ("code", "Fix my script:\n```python\nfrom openai import OpenAI\nclient = OpenAI(api_key=\"[[SECRET|{openai}]]\")\nprint(client.models.list())\n```"),
    ("code", "git push fails with: remote: Invalid username or password. My remote is https://[[SECRET|{user}:{ghp}]]@github.com/acme/app.git"),
    ("code", "Pourquoi ce curl renvoie 401 ?\ncurl -H \"Authorization: Bearer [[SECRET|{jwt}]]\" https://api.example.com/v1/me"),
    ("code", "My config.yaml:\n```yaml\ndatabase:\n  host: db.internal\n  user: app\n  password: [[SECRET|{pwd}]]\n```\nwhy can't it connect?"),
    ("code", "Is this key format right?\n[[SECRET|{pkey}]]"),
    ("code", "Traceback (most recent call last):\n  File \"[[USERNAME|C:\\Users\\{user}]]\\projets\\app\\main.py\", line 12, in <module>\n    import pandas\nModuleNotFoundError: No module named 'pandas'"),
    ("code", "Error: ENOENT: no such file or directory, open '/home/[[USERNAME|{user}]]/work/site/.env'"),
    ("code", "export GITHUB_TOKEN=[[SECRET|{ghp}]]\nwhy does gh still ask me to log in?"),
    ("code", "Mon .env :\nSTRIPE_KEY=[[SECRET|{stripe}]]\nDEBUG=false\nça marche en local mais pas en prod."),
    ("code", "Le token JWT [[SECRET|{jwt}]] est-il expiré ?"),
    ("code", "const HF_TOKEN = \"[[SECRET|{hf}]]\"; // why 403 on private repo?"),
    ("code", "SELECT * FROM users WHERE email = '[[EMAIL|{email}]]'; -- returns nothing, why?"),
    ("code", "ssh [[USERNAME|{user}]]@[[IP|{ip}]] -p 2222 hangs, any idea?"),
]

NEGATIVE = [
    ("fr", "Quelle est la capitale de l'Australie ?"),
    ("fr", "Explique-moi la différence entre un ETF et une action."),
    ("fr", "Résume la biographie de {public}."),
    ("fr", "Que pensait {public} de l'éducation ?"),
    ("fr", "Écris un poème sur l'automne à Paris."),
    ("fr", "Comment calculer la clé de contrôle d'un IBAN ? Donne la formule générale."),
    ("fr", "Le numéro de commande {badcard} ne marche pas sur le site, que faire ?"),
    ("fr", "Donne-moi une recette de tarte aux pommes pour 6 personnes."),
    ("fr", "Combien de jours entre le 12/03/2024 et le 01/09/2024 ?"),
    ("fr", "La rue de la Paix est-elle la case la plus chère du Monopoly ?"),
    ("fr", "Comment configurer une adresse e-mail professionnelle avec un nom de domaine ?"),
    ("fr", "Mon mot de passe doit contenir combien de caractères pour être sûr ?"),
    ("fr", "Traduis : « Le chat dort sur le canapé depuis midi. »"),
    ("fr", "Quel est l'indicatif téléphonique de la Belgique ?"),
    ("fr", "Le numéro SIRET a combien de chiffres ?"),
    ("en", "What did {public} contribute to computer science?"),
    ("en", "Explain how a Luhn checksum works, with a short example."),
    ("en", "Write a haiku about rain in the city."),
    ("en", "How do I reset a router to factory settings?"),
    ("en", "What is the population of Lyon?"),
    ("en", "Version 1.2.3.4 broke the build, how do I pin 1.2.3.3 with pip?"),
    ("code", "```python\nimport os\nclient = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n```\nIs reading the key from the environment the right way?"),
    ("code", "api_key = \"YOUR_API_KEY\"  # replace me\nWhy is this placeholder bad practice?"),
    ("code", "The AWS docs use AKIAIOSFODNN7EXAMPLE as a sample access key; where do I put my real one?"),
    ("code", "commit {sha} broke the tests, how do I bisect?"),
    ("code", "Generate a UUID like {uuid} in Rust."),
    ("code", "```js\nconst password = process.env.DB_PASSWORD;\nif (!password) throw new Error('missing');\n```"),
    ("code", "def luhn(n: str) -> bool:\n    s = 0\n    for i, d in enumerate(reversed(n)):\n        x = int(d) * (2 if i % 2 else 1)\n        s += x - 9 if x > 9 else x\n    return s % 10 == 0"),
    ("code", "sha256 of the file is {hex64}, how do I verify it on Windows?"),
    ("code", "docker run -p 8080:80 -v /home/user/data:/data nginx: the volume is empty"),
    ("code", "Mon chemin est C:\\Users\\Public\\Documents\\rapport.docx, comment l'ouvrir en Python ?"),
    ("code", "token = request.headers.get('Authorization', '').removeprefix('Bearer ')\nis this safe?"),
    ("code", "Explain this regex: ^[\\w.+-]+@[\\w-]+\\.[\\w.]+$"),
    ("code", "Write a SQL query that counts users per country from table customers(id, name, email, country)."),
    ("code", "base64 of 'hello world' is aGVsbG8gd29ybGQ=, right?"),
    ("code", "Listen on 0.0.0.0:8104 or 127.0.0.1:8104 for a local service?"),
    ("code", "password: ${DB_PASSWORD}\nuser: ${DB_USER}\nIs this docker-compose interpolation correct?"),
    ("code", "Set PYTHONPATH=/usr/local/lib/python3.11/site-packages and restart."),
]

MARK = re.compile(r"\[\[([A-Z_]+)\|(.*?)\]\]", re.S)
FIELD = re.compile(r"\{([a-z_0-9]+)\}")


def _values(rng: random.Random) -> dict:
    fn, ln = rng.choice(FIRST), rng.choice(LAST)
    cp, city = rng.choice(CITIES)
    user = rng.choice([fn.lower().replace("é", "e").replace("è", "e").replace("ï", "i"),
                       (fn[0] + ln).lower().replace("è", "e").replace("é", "e")])
    return dict(
        fn=fn, ln=ln, fn2=rng.choice(FIRST), ln2=rng.choice(LAST), fne=rng.choice(FIRST_EN), lne=rng.choice(LAST_EN),
        email=f"{fn.lower()}.{ln.lower()}@{rng.choice(DOMAINS)}".replace("é", "e").replace("è", "e").replace("ï", "i"),
        email2=f"{rng.choice(FIRST).lower()}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}".replace("é", "e").replace("è", "e").replace("ï", "i"),
        phone=phone_fr(rng), phone_intl=phone_intl(rng), num=rng.randint(1, 180), street=rng.choice(STREETS),
        street_en=rng.choice(STREETS_EN), cp=cp, city=city, iban=iban_fr(rng), card=card(rng), nir=nir(rng),
        dob=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2015)}",
        pwd=rng.choice(["Soleil", "Chaton", "Bordeaux", "Azerty", "Dragon"]) + str(rng.randint(10, 9999)) + rng.choice("!?#$*"),
        user=user, aws="AKIA" + rand_token(rng, 16, string.ascii_uppercase + string.digits),
        aws_secret=rand_token(rng, 40, string.ascii_letters + string.digits + "+/"),
        openai="sk-proj-" + rand_token(rng, 48, string.ascii_letters + string.digits + "_-"),
        ghp="ghp_" + rand_token(rng, 36), stripe="sk_live_" + rand_token(rng, 24), hf="hf_" + rand_token(rng, 34),
        jwt=jwt(rng), pkey=private_key(rng), ip=f"{rng.randint(11, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        public=rng.choice(PUBLIC), badcard=bad_card(rng), sha=rand_token(rng, 40, "0123456789abcdef"),
        uuid="-".join(rand_token(rng, n, "0123456789abcdef") for n in (8, 4, 4, 4, 12)),
        hex64=rand_token(rng, 64, "0123456789abcdef"),
    )


def render(template: str, values: dict) -> tuple[str, list[dict]]:
    """Fill the template, strip the markup and return (text, gold spans)."""
    filled = FIELD.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)
    text, spans, pos = [], [], 0
    out_len = 0
    for m in MARK.finditer(filled):
        before = filled[pos:m.start()]
        text.append(before)
        out_len += len(before)
        val = m.group(2)
        spans.append({"start": out_len, "end": out_len + len(val), "type": m.group(1)})
        text.append(val)
        out_len += len(val)
        pos = m.end()
    text.append(filled[pos:])
    return "".join(text), spans


def build(per_template: int = 4, seed: int = 20261009) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for label, pool in ((1, POSITIVE), (0, NEGATIVE)):
        for ti, (lang, tpl) in enumerate(pool):
            for k in range(per_template):
                text, spans = render(tpl, _values(rng))
                tid = f"hand-{'p' if label else 'n'}{ti:02d}"
                # the renderings of one template share a group: they land in the same split
                docs.append({"id": f"{tid}-{k}", "group": tid, "source": "handmade", "lang": lang,
                             "text": text, "spans": spans})
    # de-duplicate templates without fields (negatives rendered identically several times)
    seen, out = set(), []
    for d in docs:
        if d["text"] not in seen:
            seen.add(d["text"])
            out.append(d)
    return out
