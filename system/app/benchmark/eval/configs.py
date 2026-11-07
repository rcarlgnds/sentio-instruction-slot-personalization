from dataclasses import dataclass

DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)


def _renorm(alpha: float, beta: float, gamma: float) -> tuple:
    total = alpha + beta + gamma
    if total <= 0:
        return (1.0, 0.0, 0.0)
    return (alpha / total, beta / total, gamma / total)


@dataclass(frozen=True)
class AblationConfig:
    key: str
    label: str
    reranker: str
    weights: tuple = None
    family: str = "main"


def _cps(key, label, alpha, beta, gamma, family="sub"):
    a, b, g = _renorm(alpha, beta, gamma)
    return AblationConfig(key, label, "cps", (a, b, g), family)


MAIN_CONFIGS = [
    AblationConfig("A", "vanilla_cosine", "cosine"),
    _cps("B", "cps_sentio", *DEFAULT_WEIGHTS, family="main"),
    AblationConfig("C", "cross_encoder", "cross_encoder"),
    AblationConfig("D", "cps_plus_reranker", "cps_then_reranker", DEFAULT_WEIGHTS),
]

EXTRA_CONFIGS = [
    AblationConfig("C2", "gte_multilingual", "gte", family="extra"),
    AblationConfig("D2", "cps_plus_gte", "cps_then_gte", DEFAULT_WEIGHTS, family="extra"),
]

SUB_CONFIGS = [
    _cps("S1", "sim_only", 1.0, 0.0, 0.0),
    _cps("S2", "sim_prox", 0.5, 0.3, 0.0),
    _cps("S3", "sim_struct", 0.5, 0.0, 0.2),
    _cps("S4", "full", *DEFAULT_WEIGHTS),
]

DILUTION_CONFIGS = [
    AblationConfig("D625", "sim_only_diluted_0.625", "cps", (0.625, 0.0, 0.0), "dilution"),
    AblationConfig("D714", "sim_only_diluted_0.714", "cps", (0.714, 0.0, 0.0), "dilution"),
    AblationConfig("D500", "sim_only_diluted_0.500", "cps", (0.500, 0.0, 0.0), "dilution"),
]

ALL_CONFIGS = MAIN_CONFIGS + SUB_CONFIGS + EXTRA_CONFIGS + DILUTION_CONFIGS
CONFIG_BY_KEY = {c.key: c for c in ALL_CONFIGS}


def get_configs(keys=None) -> list:
    if not keys:
        return list(MAIN_CONFIGS)
    return [CONFIG_BY_KEY[k] for k in keys]
