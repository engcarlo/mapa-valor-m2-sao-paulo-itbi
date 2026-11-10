"""O loader precisa sobreviver aos defeitos reais dos arquivos da Prefeitura."""
import numpy as np
import pandas as pd
import pytest

from itbi_sp.carregamento import (carregar_arquivo_anual, formatar_cep, normalizar_acc,
                                  normalizar_padrao)
from itbi_sp.config import COL_ACC, COL_DESC_PADRAO, COLUNAS_PADRAO


def _linha(acc=2009, desc_padrao="RESIDENCIAL VERTICAL", padrao=22, cep=4516000):
    v = {c: None for c in COLUNAS_PADRAO}
    v.update({"N° do Cadastro (SQL)": 1234, "CEP": cep, "Uso (IPTU)": 20, "Padrão (IPTU)": padrao,
              "Natureza de Transação": "1.Compra e venda"})
    return [v[c] for c in COLUNAS_PADRAO[:-2]] + [desc_padrao, acc]


def _salvar(tmp_path, abas: dict):
    caminho = tmp_path / "itbi.xlsx"
    with pd.ExcelWriter(caminho) as xw:
        for nome, df in abas.items():
            df.to_excel(xw, sheet_name=nome, index=False, header=df.attrs.get("header", True))
    return str(caminho)


def test_cabecalho_mal_rotulado_preserva_ano_de_construcao(tmp_path):
    """Regressão do bug: em 2019-2022/2024 as duas últimas colunas chegam como
    'ACC (IPTU)' e 'ACC (IPTU).1'; a última é o ANO e antes era descartada."""
    cols = COLUNAS_PADRAO[:-2] + ["ACC (IPTU)", "ACC (IPTU)"]
    df = pd.DataFrame([_linha(acc=2009), _linha(acc=1984)], columns=cols)
    arq = _salvar(tmp_path, {"JAN-2022": df})
    d = carregar_arquivo_anual(arq, 2022)
    assert d[COL_ACC].tolist() == [2009, 1984]
    assert (d[COL_DESC_PADRAO] == "RESIDENCIAL VERTICAL").all()


def test_aba_sem_cabecalho(tmp_path):
    df = pd.DataFrame([_linha(acc=1939), _linha(acc=2008)])
    df.attrs["header"] = False
    arq = _salvar(tmp_path, {"JAN-2024": df})
    d = carregar_arquivo_anual(arq, 2024)
    assert d[COL_ACC].tolist() == [1939, 2008]
    assert list(d.columns[: len(COLUNAS_PADRAO)]) == COLUNAS_PADRAO


def test_aba_com_27_colunas_completa_com_na(tmp_path):
    df = pd.DataFrame([_linha()]).iloc[:, :27]
    df.attrs["header"] = False
    arq = _salvar(tmp_path, {"OUT-2024": df})
    d = carregar_arquivo_anual(arq, 2024)
    assert d[COL_ACC].isna().all() and len(d) == 1


def test_coluna_excedente_quase_vazia_e_descartada_mas_com_dado_real_falha(tmp_path):
    cols = COLUNAS_PADRAO + ["lixo"]
    df = pd.DataFrame([_linha() + [None] for _ in range(20)], columns=cols)
    df.loc[0, "lixo"] = "x"
    arq = _salvar(tmp_path, {"MAR-2008": df})
    assert len(carregar_arquivo_anual(arq, 2008)) == 20
    df["lixo"] = "dado"
    arq2 = _salvar(tmp_path, {"MAR-2008": df})
    with pytest.raises(ValueError):
        carregar_arquivo_anual(arq2, 2008)


def test_filtro_aplicado_por_aba(tmp_path):
    a = pd.DataFrame([_linha(), _linha()], columns=COLUNAS_PADRAO)
    arq = _salvar(tmp_path, {"JAN-2022": a, "FEV-2022": a, "LEGENDA": pd.DataFrame({"x": [1]})})
    d = carregar_arquivo_anual(arq, 2022, filtro=lambda x: x.iloc[:1])
    assert len(d) == 2 and set(d["mes_planilha"]) == {"JAN-2022", "FEV-2022"}


def test_formatar_cep_restaura_zero():
    s = formatar_cep(pd.Series([4516000, 1310100, np.nan, "01310-100"], dtype=object))
    assert s.iloc[0] == "04516000" and s.iloc[1] == "01310100"
    assert pd.isna(s.iloc[2])


def test_normalizar_padrao_float_e_string():
    assert normalizar_padrao(pd.Series([22, 22.0, "22", "23.0"])).tolist() == ["22", "22", "22", "23"]


def test_normalizar_acc_descarta_invalidos():
    acc = pd.Series([2009, 0, 1850, 2030, np.nan, 2024])
    ano = pd.Series([2020] * 6)
    r = normalizar_acc(acc, ano)
    assert r.tolist()[0] == 2009 and r.isna().tolist() == [False, True, True, True, True, False]
