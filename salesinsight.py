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
        #Vamos substituir todos os traços baixos por espaços normais.
        nome_formatado = chave.replace("_", " ").capitalize()
        print(f"  {nome_formatado}: {valor}")
    print("============================\n")

    return df, relatorio

def criar_colunas_derivadas(df_limpo):
    """RF04 – Criar Colunas Derivadas com Transformações."""
    # Criamos uma cópia para trabalhar com segurança nas novas colunas
    df = df_limpo.copy()

    # Receita total por linha de venda (quantidade * preco_unitario)
    df["receita_total"] = df["quantidade"] * df["preco_unitario"]

    # Extração de componentes de data
    df["mes"] = df["data_venda"].dt.month
    df["trimestre"] = df["data_venda"].dt.quarter.apply(lambda q: f"Q{q}")
    df["ano"] = df["data_venda"].dt.year

    # Mapeamento para garantir que o nome do mês saia sempre em Português
    meses_pt = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    df["mes_nome"] = df["mes"].map(meses_pt)

    # Classificação da receita por item com numpy.select
    condicoes = [
        df["receita_total"] < 500,
        (df["receita_total"] >= 500) & (df["receita_total"] < 5000),
        df["receita_total"] >= 5000
    ]
    classificacoes = ["Baixo Valor", "Médio Valor", "Alto Valor"]
    df["faixa_receita_item"] = np.select(condicoes, classificacoes, default="Não Classificado")

    print("=== COLUNAS DERIVADAS CRIADAS ===")
    print(df[["data_venda", "receita_total", "mes_nome", "trimestre", "faixa_receita_item"]].head())
    print("=================================\n")

    return df


# BLOCO PRINCIPAL
if __name__ == "__main__":
    path = "vendas.csv"
    
    try:
        # Carregamos os dados brutos do csv
        df_vendas = pd.read_csv(path)
        
        # Vamos executar a inspeção e leitura dos dados (RF02)
        inspecionar_dados(df_vendas)
        
        # Vamos executar a limpeza (RF03)
        df_limpo, relatorio_limpeza = limpar_dados(df_vendas)
        
        # Vamos executar o enriquecimento (RF04)
        df_enriquecido = criar_colunas_derivadas(df_limpo)
        
        # Vamos salvar o resultado final processado
        df_enriquecido.to_csv("vendas_limpo.csv", index=False)
        print("[Sucesso] Pipeline executado com sucesso! Arquivo 'vendas_limpo.csv' gerado.")
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{path}' não foi encontrado na raiz do projeto.")