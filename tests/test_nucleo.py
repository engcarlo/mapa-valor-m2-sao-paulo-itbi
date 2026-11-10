import numpy as np
import pandas as pd
import pytest

from itbi_sp import ml
from itbi_sp.geo import atribuir_distrito_pontos, carregar_poligonos_distrito
from itbi_sp.mapa import calcular_stats_por_ano, montar_stats_data
from itbi_sp.temporal import aplicar_correcao_temporal


def test_correcao_temporal_formula_e_descarte():
    serie = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    df = pd.DataFrame({"data": pd.to_datetime(["2020-01-15", "2020-03-01", "1990-01-01"]), "v": [1000.0] * 3})
    r = aplicar_correcao_temporal(df, serie, "data", "v")
    assert len(r) == 2                                            # 1990 fora do range
    assert r.loc[r.index[0], "valor_corrigido"] == pytest.approx(1210.0)   # 121/100
    assert r.loc[r.index[1], "valor_corrigido"] == pytest.approx(1000.0)   # já na data-base


def test_distrito_vetorizado_praca_da_se():
    p, n, t = carregar_poligonos_distrito()
    r = atribuir_distrito_pontos([-23.5505, np.nan, 0.0], [-46.6333, -46.6, 0.0], p, n, t)
    assert r[0] == "SE" and r[1] is None and r[2] is None
    assert len(n) == 96


def test_stats_por_ano_aceita_padrao_float_e_recombinacao_exata():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"preco": rng.lognormal(8.5, .3, 600), "distrito": "SE", "padrao": [22.0] * 600,
                      "ano": np.repeat([2019, 2020, 2021], 200)})
    st = calcular_stats_por_ano(d, "preco", "distrito", "padrao", "ano")
    assert set(st["padrao"]) == {"22"} and st["n"].sum() == 600
    # variância combinada reproduz o desvio do período inteiro (base do slider)
    n, m, s = st["n"].to_numpy(), st["media"].to_numpy(), st["desvio"].to_numpy()
    media = (n * m).sum() / n.sum()
    var = ((n - 1) * s**2 + n * (m - media) ** 2).sum() / (n.sum() - 1)
    assert np.sqrt(var) == pytest.approx(d["preco"].std(), rel=1e-9)
    assert montar_stats_data(st)["22"]["SE"]["2019"][0] == 200


def test_split_temporal_sem_sobreposicao():
    d = pd.DataFrame({"mes_transacao": pd.date_range("2019-01-01", periods=72, freq="MS")})
    tr, te = ml.split_temporal(d, "2024-01-01", inicio_treino="2020-01-01")
    assert tr.mes_transacao.max() < te.mes_transacao.min() and tr.mes_transacao.min() == pd.Timestamp("2020-01-01")


def test_conformal_cobertura_proxima_do_alvo():
    rng = np.random.default_rng(1)
    pad = rng.choice(ml.PADROES_RESIDENCIAL_VERTICAL, 20000)
    ruido = rng.normal(0, 0.2, 20000) * (1 + (pad == "25"))
    y, p = 10 + ruido, np.full(20000, 10.0)
    ic = ml.IntervaloConformal(alpha=0.2).ajustar(y[:10000], p[:10000], pad[:10000])
    cob = ic.cobertura(y[10000:], p[10000:], pad[10000:]).set_index("padrao")["cobertura_%"]
    assert abs(cob["GLOBAL"] - 80) < 2 and abs(cob["25"] - 80) < 5


def test_corrigir_vies_remove_deslocamento():
    rng = np.random.default_rng(2)
    pad = np.array(["22"] * 1000)
    y = rng.normal(10, .1, 1000)
    pc, pa = ml.corrigir_vies(y, y - 0.05, pad, y - 0.05, pad)
    assert np.median(y - pa) == pytest.approx(0.0, abs=1e-9)


def test_metricas_por_padrao_tem_global_e_classes():
    y = np.log(np.array([100., 200., 300., 400.])); p = y + np.log(1.1)
    r = ml.metricas_por_padrao(y, p, ["22", "22", "23", "23"])
    assert r.iloc[0]["padrao"] == "GLOBAL" and r.iloc[0]["mdape_%"] == pytest.approx(10.0)
    assert set(r["padrao"]) == {"GLOBAL", "22", "23"}


def test_montar_features_idade_e_tempo():
    d = pd.DataFrame({"Área Construída (m2)": [80.0], "valor_corrigido": [4e5], "Padrão (IPTU)": ["22"],
                      "ano_transacao": [2020], "acc_valido": [2010.0],
                      "mes_transacao": [pd.Timestamp("2007-03-01")], "Área do Terreno (m2)": [1000.0],
                      "Fração Ideal": [0.01], "CEP": ["04516000"]})
    f = ml.montar_features(d)
    assert f.loc[0, "idade"] == 10 and f.loc[0, "meses"] == 14
    assert f.loc[0, "log_terreno_ideal"] == pytest.approx(np.log(10.0))


def test_preparar_apartamentos_converte_numericos_vindos_do_cache():
    from itbi_sp.config import COLUNAS_PADRAO
    from itbi_sp.depuracao import preparar_apartamentos
    d = pd.DataFrame({c: [None, None] for c in COLUNAS_PADRAO}).astype(object)
    d["Valor de Transação (declarado pelo contribuinte)"] = ["400000", "100"]     # 2ª = simbólico
    d["Área Construída (m2)"] = ["80", "50"]
    d["Valor Venal de Referência (proporcional)"] = ["250000.5", "1"]
    d["Data de Transação"] = ["2021-05-03", "2021-05-04"]
    d["CEP"] = ["4516000", "4516000"]; d["Padrão (IPTU)"] = ["22.0", "22.0"]; d["ACC (IPTU)"] = ["2010", "2010"]
    r = preparar_apartamentos(d)
    assert len(r) == 1 and r["Valor Venal de Referência (proporcional)"].iloc[0] == 250000.5
    assert r["CEP"].iloc[0] == "04516000" and r["Padrão (IPTU)"].iloc[0] == "22" and r["acc_valido"].iloc[0] == 2010
