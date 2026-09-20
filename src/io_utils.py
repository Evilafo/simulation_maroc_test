"""Point d'entree modulaire : chargement de la configuration, des donnees
et journalisation de l'environnement d'execution (Etape 1, UC-S1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resout un chemin de configs.paths en objet Path (relatif au projet
    si le chemin n'est pas absolu)."""
    raw = cfg["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _local_data_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / filename


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(column).lstrip("\ufeff") for column in df.columns]
    return df


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logging(cfg: dict[str, Any]) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    log_file = PROJECT_ROOT / log_cfg.get("log_file", "outputs/run_log.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_cfg.get("level", "INFO"))

    logger = logging.getLogger("uc_s1_etape1")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def log_environment(logger: logging.Logger) -> dict[str, Any]:
    """Journalise les versions de l'environnement Python/paquets utilises."""
    import matplotlib
    import numpy as np
    import scipy
    import sklearn
    import statsmodels

    env = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "statsmodels": statsmodels.__version__,
    }
    logger.info("Environnement d'execution : %s", json.dumps(env, ensure_ascii=False))
    return env


def load_work_base(cfg: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Charge base_MEF_1991_2024_9pays_28indicateurs.csv (fichier de travail)."""
    path = resolve_path(cfg, "work_base_mef")
    if not path.exists():
        source_path = _local_data_path("base_B_revised_wide.csv")
        if not source_path.exists():
            raise FileNotFoundError(
                f"Fichier de travail introuvable : {path}. "
                "Verifiez configs/config.yaml -> paths.work_base_mef."
            )
        source = _read_csv(source_path)
        excluded = {"DETTE_PUBLIQUE", "TCER"}
        columns = [column for column in source.columns if column not in excluded]
        df = source[columns].copy()
        df = df[df["year"].between(1991, 2024)].reset_index(drop=True)
        path = source_path
    else:
        df = _read_csv(path)
    if logger:
        logger.info(
            "Fichier de travail charge : %s (shape=%s, sha256=%s)",
            path, df.shape, sha256_of_file(path)[:16],
        )
    return df


def load_source_base(cfg: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Charge base_B_revised_wide.csv (base source, 1980-2024, 30 indicateurs)."""
    path = resolve_path(cfg, "source_base_B_revised_wide")
    if not path.exists():
        path = _local_data_path("base_B_revised_wide.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"Base source introuvable : {path}. "
                "Verifiez configs/config.yaml -> paths.source_base_B_revised_wide."
            )
    df = _read_csv(path)
    if logger:
        logger.info(
            "Base source chargee : %s (shape=%s, sha256=%s)",
            path, df.shape, sha256_of_file(path)[:16],
        )
    return df


def load_variable_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(cfg, "variable_registry")
    if not path.exists():
        raise FileNotFoundError(f"Registre des variables introuvable : {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def indicator_columns(df: pd.DataFrame, id_cols=("country_iso3", "year")) -> list[str]:
    return [c for c in df.columns if c not in id_cols]
