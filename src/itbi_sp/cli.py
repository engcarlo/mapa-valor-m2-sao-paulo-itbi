"""Interface de linha de comando: ``itbi-sp {base,mapa,diagnostico,ml} --config config.toml``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="itbi-sp", description=__doc__)
    ap.add_argument("comando", choices=["base", "mapa", "diagnostico", "ml"])
    ap.add_argument("--config", required=True, help="arquivo TOML (ver config.example.toml)")
    ap.add_argument("--sem-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")

    cfg = pipeline.carregar_configuracao(a.config)
    saida = Path(cfg["caminhos"]["saida_dir"])
    saida.mkdir(parents=True, exist_ok=True)
    base = pipeline.construir_base(cfg, usar_cache=not a.sem_cache)
    print(f"base: {len(base):,} apartamentos, {base['ano_arquivo'].min()}-{base['ano_arquivo'].max()}")

    if a.comando == "base":
        base.to_parquet(saida / "base_apartamentos.parquet")
    elif a.comando == "mapa":
        print("mapa:", pipeline.gerar_mapa(base, str(saida / "mapa_preco_m2_distrito_sp.html")))
    elif a.comando == "diagnostico":
        por_era, pool = pipeline.diagnostico_temporal(base)
        print(por_era.to_string(index=False), "\n", pool.to_string(index=False))
    elif a.comando == "ml":
        from .experimento_ml import rodar_experimento
        resumo, por_padrao, cobertura, _ = rodar_experimento(base, **cfg.get("ml", {}))
        resumo.to_csv(saida / "ml_resumo.csv", index=False)
        por_padrao.to_csv(saida / "ml_por_padrao.csv", index=False)
        cobertura.to_csv(saida / "ml_cobertura.csv", index=False)
        pd.set_option("display.width", 220)
        print(resumo.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
