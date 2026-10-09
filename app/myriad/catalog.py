"""Curated catalogue of small open-weight models (Apache-2.0 or MIT only) in GGUF, with pinned
revisions, sizes and SHA-256 digests (taken from the Hugging Face API), and the recommendation used by
the first-run wizard.

Several families matter more than the best single model: the swarm fuses answers from different
families, so the recommendation also nudges towards families the network is short of."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .hardware import Hardware, estimate_tps, model_budget_gb

PERMISSIVE = {"apache-2.0", "mit"}
QUANTS = ("Q4_K_M", "Q8_0")
DEFAULT_QUANT = "Q4_K_M"
MIN_USEFUL_TPS = 4.0  # below this a peer is too slow to be useful in a vote (answers time out)


@dataclass(frozen=True)
class ModelFile:
    file: str
    size: int  # bytes
    sha256: str


@dataclass(frozen=True)
class CatalogModel:
    id: str  # stable key used by the UI and the config
    name: str
    family: str
    params_b: float
    licence: str
    repo: str
    revision: str  # pinned commit of the repository
    files: dict  # quant -> ModelFile
    blurb_fr: str
    blurb_en: str
    gsm8k: float | None = None  # measured accuracy in phase 0 when known

    def file_for(self, quant: str) -> ModelFile:
        if quant not in self.files:
            raise KeyError(f"quantification {quant} indisponible pour {self.id}")
        return self.files[quant]

    def spec(self, quant: str) -> str:
        """'repo:file.gguf', the format of Config.model."""
        return f"{self.repo}:{self.file_for(quant).file}"

    def url(self, quant: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file_for(quant).file}"


def _m(**kw) -> CatalogModel:
    kw["files"] = {q: ModelFile(*v) for q, v in kw["files"].items()}
    return CatalogModel(**kw)


CATALOG: tuple[CatalogModel, ...] = (
    _m(id="qwen3.5-2b", name="Qwen3.5 2B", family="qwen", params_b=2.0, licence="apache-2.0",
       repo="unsloth/Qwen3.5-2B-GGUF", revision="f6d5376be1edb4d416d56da11e5397a961aca8ae",
       files={"Q4_K_M": ("Qwen3.5-2B-Q4_K_M.gguf", 1280835840,
                         "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"),
              "Q8_0": ("Qwen3.5-2B-Q8_0.gguf", 2012012800,
                       "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1")},
       blurb_fr="Très léger et rapide, bon partout.", blurb_en="Very light and fast, a good all-rounder."),
    _m(id="qwen3.5-4b", name="Qwen3.5 4B", family="qwen", params_b=4.0, licence="apache-2.0",
       repo="unsloth/Qwen3.5-4B-GGUF", revision="e87f176479d0855a907a41277aca2f8ee7a09523",
       files={"Q4_K_M": ("Qwen3.5-4B-Q4_K_M.gguf", 2740937888,
                         "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
              "Q8_0": ("Qwen3.5-4B-Q8_0.gguf", 4482403488,
                       "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1")},
       blurb_fr="Le plus fort de sa taille en calcul et en raisonnement.",
       blurb_en="The strongest of its size at maths and reasoning."),
    _m(id="gemma-4-e2b", name="Gemma 4 E2B", family="gemma", params_b=2.0, licence="apache-2.0",
       repo="unsloth/gemma-4-E2B-it-GGUF", revision="0314792d7f1f7e229411f620751375812bb9faf2",
       files={"Q4_K_M": ("gemma-4-E2B-it-Q4_K_M.gguf", 3106738272,
                         "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"),
              "Q8_0": ("gemma-4-E2B-it-Q8_0.gguf", 5048352864,
                       "605d3c2647d7c58c1e4b5375ccb5702acf94c2611b4c8d4877812f8fdd32d053")},
       blurb_fr="Google, multilingue, 2 milliards de paramètres effectifs.",
       blurb_en="Google, multilingual, 2B effective parameters.", gsm8k=0.74),
    _m(id="gemma-4-e4b", name="Gemma 4 E4B", family="gemma", params_b=4.0, licence="apache-2.0",
       repo="unsloth/gemma-4-E4B-it-GGUF", revision="bfc15c382204943c3a8fff0c750b94ae2364d7a3",
       files={"Q4_K_M": ("gemma-4-E4B-it-Q4_K_M.gguf", 4977171584,
                         "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87"),
              "Q8_0": ("gemma-4-E4B-it-Q8_0.gguf", 8192953472,
                       "f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012")},
       blurb_fr="Le grand frère de E2B, pour les GPU de 8 Go et plus.",
       blurb_en="E2B's big sibling, for GPUs with 8 GB or more."),
    _m(id="granite-4.2-3b", name="Granite 4.2 3B", family="granite", params_b=3.0, licence="apache-2.0",
       repo="ibm-granite/granite-4.2-3b-GGUF", revision="c40945d71cd90f249a56985e8155551a9188dc30",
       files={"Q4_K_M": ("granite-4.2-3b-Q4_K_M.gguf", 2244011552,
                         "e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5"),
              "Q8_0": ("granite-4.2-3b-Q8_0.gguf", 3892651552,
                       "fbe986738041418e26de9e123ba740cb654931f85bf572a71bd01f9e6b85e53d")},
       blurb_fr="IBM, sobre et fiable, une famille de plus pour le vote.",
       blurb_en="IBM, sober and reliable, one more family for the vote."),
    _m(id="smollm3-3b", name="SmolLM3 3B", family="smollm", params_b=3.1, licence="apache-2.0",
       repo="ggml-org/SmolLM3-3B-GGUF", revision="4965cb60b150737b68a0408c36aeefb65078f894",
       files={"Q4_K_M": ("SmolLM3-Q4_K_M.gguf", 1915305312,
                         "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e"),
              "Q8_0": ("SmolLM3-Q8_0.gguf", 3275574624,
                       "8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248")},
       blurb_fr="Hugging Face, entièrement ouvert, le meilleur du réseau en phase 0.",
       blurb_en="Hugging Face, fully open, the best of the network in phase 0.", gsm8k=0.865),
    _m(id="ministral-3-3b", name="Ministral 3 3B", family="mistral", params_b=3.4, licence="apache-2.0",
       repo="mistralai/Ministral-3-3B-Instruct-2512-GGUF", revision="eb599d408350ea2bb60452cb86be7c7b2fc28227",
       files={"Q4_K_M": ("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", 2147023008,
                         "9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8"),
              "Q8_0": ("Ministral-3-3B-Instruct-2512-Q8_0.gguf", 3652204704,
                       "8c2b72eb5861304fcfd5e82f1eddd6efa4115737f4239fd216a028a8852413ef")},
       blurb_fr="Mistral AI, à l'aise en français.", blurb_en="Mistral AI, at ease in French."),
    _m(id="phi-4-mini", name="Phi-4 mini", family="phi", params_b=3.8, licence="mit",
       repo="unsloth/Phi-4-mini-instruct-GGUF", revision="78eb92a46fc37e6b524df991ed9aca9bc6aa7b80",
       files={"Q4_K_M": ("Phi-4-mini-instruct-Q4_K_M.gguf", 2491874272,
                         "88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a"),
              "Q8_0": ("Phi-4-mini-instruct.Q8_0.gguf", 4084611040,
                       "26188c6050d525376a88b04514c236c5e28a36730f1e936f2a00314212b7ba42")},
       blurb_fr="Microsoft, licence MIT, solide en raisonnement.",
       blurb_en="Microsoft, MIT licence, solid at reasoning."),
)

assert all(m.licence in PERMISSIVE for m in CATALOG)
BY_ID = {m.id: m for m in CATALOG}


def get(model_id: str) -> CatalogModel:
    try:
        return BY_ID[model_id]
    except KeyError:
        raise KeyError(f"modèle inconnu : {model_id}") from None


def fits(size_bytes: int, budget_gb: float) -> bool:
    """Weights plus ~15 % for the context and compute buffers must fit in the budget."""
    return size_bytes * 1.15 / 1024**3 <= budget_gb


def evaluate(hw: Hardware, network_families: dict[str, int] | None = None) -> list[dict]:
    """Each catalogue entry with, per quantisation: size, fit, expected tokens/s."""
    budget = model_budget_gb(hw)
    out = []
    for m in CATALOG:
        quants = {}
        for q, f in m.files.items():
            gb = f.size / 1024**3
            quants[q] = {"file": f.file, "size": f.size, "size_gb": round(gb, 2), "fits": fits(f.size, budget),
                         "tps": round(estimate_tps(hw, gb), 1)}
        d = asdict(m)
        d.pop("files")
        d.update(quants=quants, peers=(network_families or {}).get(m.family, 0))
        out.append(d)
    return out


def recommend(hw: Hardware, network_families: dict[str, int] | None = None) -> dict:
    """Pick (model, quant): among the models that fit at Q4_K_M and reach a useful speed, the largest
    one (quality), with a bonus for a family the network lacks; Q8_0 when it fits with room to spare
    and stays fast. Falls back to the smallest model when nothing is comfortable."""
    budget = model_budget_gb(hw)
    fam = network_families or {}
    best, best_score = None, None
    for m in CATALOG:
        f = m.file_for(DEFAULT_QUANT)
        gb = f.size / 1024**3
        tps = estimate_tps(hw, gb)
        if not fits(f.size, budget) or tps < MIN_USEFUL_TPS:
            continue
        rarity = 1.0 / (1 + fam.get(m.family, 0))  # 1 for a family absent from the network
        score = m.params_b * (1.0 + 0.5 * rarity) + (0.3 if m.gsm8k else 0.0) + min(tps, 60) / 100
        if best_score is None or score > best_score:
            best, best_score = m, score
    reason = "fits"
    if best is None:
        best, reason = min(CATALOG, key=lambda m: m.file_for(DEFAULT_QUANT).size), "smallest"
    quant = DEFAULT_QUANT
    q8 = best.files.get("Q8_0")
    if q8 and fits(q8.size, budget * 0.7) and estimate_tps(hw, q8.size / 1024**3) >= 25:
        quant = "Q8_0"
    return {"model": best.id, "quant": quant, "reason": reason, "budget_gb": round(budget, 2)}
