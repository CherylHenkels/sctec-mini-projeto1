import pandas as pd
import numpy as np

def inspecionar_dados(df):
    """RF02 – Inspecionar e Descrever os Dados."""
    print("=== INSPEÇÃO INICIAL DO DATASET ===")
    
    # Dimensão da tabela
    print(f"Shape: {df.shape}")
    
    # Lista de Colunas
    print(f"Colunas: {list(df.columns)}")
    
    # Tipos de dados
    print("\nTipos de dados:")
    print(df.dtypes)
    
    # Valores nulos por coluna
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    
    # Primeiros registros
    print("===================================\n")
    print("Primeiros registros:")
    print(df.head())
    print("\n")


def limpar_dados(df_bruto):
    """RF03 – Limpar e Tratar os Dados."""

    # Vamos criar uma cópia dos dados originais para protegê-los
    df = df_bruto.copy()
    n_inicial = len(df)
    relatorio = {}

    # Vamos remover possíveis colunas duplicadas pelo ID da venda
    n_antes_dup = n_inicial
    df = df.drop_duplicates(subset="id_venda")
    relatorio["duplicatas_removidas"] = n_antes_dup - len(df)

    # Vamos converter data e remover dados/strings inválidas (ex: "DATA INVÁLIDA")
    df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
    n_datas_invalidas = df["data_venda"].isnull().sum()
    df = df.dropna(subset=["data_venda"])
    relatorio["datas_invalidas_removidas"] = n_datas_invalidas

    # Vamos remover linhas com quantidade ou preço nulos
    n_antes_nan = len(df)
    df = df.dropna(subset=["quantidade", "preco_unitario", "produto", "cliente"])
    relatorio["linhas_nulas_removidas"] = n_antes_nan - len(df)

    # Vamos filtrar para que quantidade e preço sejam positivos
    n_antes_inv = len(df)
    df = df[(df["quantidade"] > 0) & (df["preco_unitario"] > 0)]
    relatorio["invalidos_removidos"] = n_antes_inv - len(df)
  
    # Vamos remover espaços e aplicar o padrão Title Case nas colunas de texto (Sem quebras)
    for col in ["categoria", "regiao", "produto", "cliente"]:
        df[col] = df[col].str.strip().str.title()

    # Vamos garantir tipos numéricos corretos (quantidade como inteiro)
    df["quantidade"] = df["quantidade"].astype(int) 

    # Métricas Finais
    n_final = len(df)
    relatorio["registros_iniciais"] = n_inicial
    relatorio["registros_finais"] = n_final
    relatorio["registros_removidos_total"] = n_inicial - n_final

    print("\n=== RELATÓRIO DE LIMPEZA ===")
    for chave, valor in relatorio.items():
        nome_formatado = chave.replace("_", " ").capitalize()
        print(f"  {nome_formatado}: {valor}")
    print("============================\n")

    return df, relatorio