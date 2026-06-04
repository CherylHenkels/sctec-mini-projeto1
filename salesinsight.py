import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def inspecionar_dados(df):
    """RF02 – Inspecionar e Descrever os Dados."""
    print("=" * 50)
    print("     INSPEÇÃO INICIAL DO DATASET")
    print("=" * 50)

    print(f"\nShape: {df.shape}")

    print(f"\nColunas: {list(df.columns)}")

    print(f"\nTipos de dados:\n{df.dtypes}")

    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

    print(f"\nPrimeiros registros:\n{df.head()}")

    print(f"\nEstatísticas descritivas:\n{df.describe()}")


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
    relatorio["datas_invalidas_removidas"] = int(n_datas_invalidas)

    # Vamos remover linhas com quantidade ou preço nulos
    n_antes_nan = len(df)
    df = df.dropna(subset=["quantidade", "preco_unitario", "produto", "cliente"])
    relatorio["linhas_nulas_removidas"] = int(n_antes_nan - len(df))

    # Vamos filtrar para que quantidade e preço sejam positivos
    n_antes_inv = len(df)
    df = df[(df["quantidade"] > 0) & (df["preco_unitario"] > 0)]
    relatorio["invalidos_removidos"] = n_antes_inv - len(df)
  
    # Vamos remover espaços e aplicar o padrão Title Case nas colunas de texto (Sem quebras)
    for col in ["categoria", "regiao", "produto", "cliente"]:
        df[col] = df[col].str.strip().str.title()

    # Vamos garantir tipos numéricos corretos (quantidade como inteiro)
    df["quantidade"] = df["quantidade"].astype(int) 
    df["preco_unitario"] = df["preco_unitario"].astype(float)

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
        1: "Janeiro",  2: "Fevereiro", 3: "Março",     4: "Abril",
        5: "Maio",     6: "Junho",     7: "Julho",     8: "Agosto",
        9: "Setembro", 10: "Outubro",  11: "Novembro", 12: "Dezembro"
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

    # Desconto por faixa de receita
    df["desconto"] = df["receita_total"].apply(lambda x: 0.10 if x > 10000 else 0.05)

    print("=== COLUNAS DERIVADAS CRIADAS ===")
    print(df[["data_venda", "receita_total", "mes_nome", "trimestre", "faixa_receita_item", "desconto"]].head())
    print("=================================\n")

    return df

def calcular_metricas(df):
    """RF05 – Calcular Métricas Agregadas (groupby)."""
    metricas = {}

    # Receita por mês
    por_mes = df.groupby("mes").agg(
        receita_total = ("receita_total", "sum"),
        quantidade    = ("quantidade",    "sum"),
        n_vendas      = ("id_venda",      "count")
    ).reset_index().sort_values("mes")
    metricas["por_mes"] = por_mes

    # Top 5 produtos por receita
    top_produtos = (
        df.groupby("produto")["receita_total"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    metricas["top_produtos"] = top_produtos

    # Receita por categoria
    por_categoria = (
        df.groupby("categoria")["receita_total"]
        .sum().
        reset_index()
    )
    metricas["por_categoria"] = por_categoria

    # Receita por região
    por_regiao = df.groupby("regiao").agg(
        receita_total  =  ("receita_total", "sum"),
        media_ticket   =  ("receita_total", "mean")
    ).reset_index().sort_values("receita_total", ascending=False)
    metricas["por_regiao"] = por_regiao

    # Receita por trimestre
    por_trimestre = df.groupby("trimestre")["receita_total"].sum().reset_index()
    metricas["por_trimestre"] = por_trimestre

    # Exibição bonita no console usando o mesmo padrão do capitalize
    print(f"====== MÉTRICAS AGREGADAS ======")
    for nome, tabela in metricas.items():
        print(f"=== {nome.upper().replace('_', ' ')} ===")
        print(tabela.to_string(index=False) + "\n")
    print("=================================\n")
      
    return metricas

def segmentar_clientes(df):
    """RF06 – Segmentar Clientes por Nível de Gasto (Versão Avançada)."""
    
      # Em vez de apenas somar, vamos calcular o gasto total, a média por Compra e o total de Vendas do cliente
    clientes = df.groupby("cliente").agg(
        total_gasto        = ("receita_total", "sum"),
        ticket_medio       = ("receita_total", "mean"),
        frequencia_compras = ("id_venda", "count")
    ).reset_index()

    # AQUI USAMOS A FUNÇÃO LAMBDA E O CONDICIONAL!: Classificação usando a função lambda com as condições exigidas
    clientes["segmento"] = clientes["total_gasto"].apply(
        lambda gasto: "Ouro" if gasto > 15000
                             else ("Prata" if gasto >= 5000 else "Bronze")
    )

    # Ordena do cliente que mais gerou receita para o que menos gerou
    clientes = clientes.sort_values("total_gasto", ascending=False)

    print("=== SEGMENTAÇÃO DE CLIENTES (RANKING TOP 10) ===")
    # Arredonda os valores float para 2 casas decimais na exibição
    print(clientes.head(10).round(2).to_string(index=False))
    
    print(f"\nDistribuição de segmentos na carteira:")
    print(clientes["segmento"].value_counts())
    print("===============================================\n")

    return clientes

def calcular_estatisticas_numpy(df):
    """RF07 – Calcular Estatísticas Avançadas com NumPy."""
    print("=== ESTATÍSTICAS MATRICIAIS COM NUMPY ===")

    #Conversão da coluna do DataFrame para array NumPy bruto
    receitas = df["receita_total"].to_numpy()

      # 2. Uso de múltiplas funções estatísticas nativas do NumPy
    media         = np.mean(receitas)
    mediana       = np.median(receitas)
    desvio_padrao = np.std(receitas)
    total         = np.sum(receitas)
    p25           = np.percentile(receitas, 25)
    p75           = np.percentile(receitas, 75)

    print(f"  Receita média por venda:    R$ {media:.2f}")
    print(f"  Receita mediana por venda:  R$ {mediana:.2f}")
    print(f"  Desvio padrão amostral:     R$ {desvio_padrao:.2f}")
    print(f"  Faturamento total acumulado:R$ {total:.2f}")
    print(f"  Percentil 25 (Q1):          R$ {p25:.2f}")
    print(f"  Percentil 75 (Q3):          R$ {p75:.2f}")

    #Demonstração de Broadcasting: normalizar vetor de receitas entre 0 e 1 de uma vez
    receitas_normalizadas = (receitas - receitas.min()) / (receitas.max() - receitas.min())
    print(f"\n  Vetor normalizado via Broadcasting (primeiros 5): {receitas_normalizadas[:5].round(4)}")

    #Operação vetorizada: filtragem condicional em blocos de memória C (sem loops 'for')
    acima_da_media = receitas[receitas > media]
    print(f"  Vendas estritamente acima da média: {len(acima_da_media)} de {len(receitas)}")
    print("=========================================\n")

    return {
        "media": media, "mediana": mediana,
        "desvio_padrao": desvio_padrao, "total": total
    }

def gerar_visualizacoes(df, metricas, output_dir="outputs/graficos"):
    """RF08 – Criar Visualizações com Matplotlib e Seaborn (5 Gráficos)."""
    os.makedirs(output_dir, exist_ok=True)
    print("=== INICIANDO EXPORTAÇÃO GRÁFICA ===")

      # Configurações estéticas globais
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

      # --- Gráfico 1: Receita por Mês (linha) ---
    fig, ax     = plt.subplots()
    por_mes = metricas["por_mes"]
    ax.plot(por_mes["mes"], por_mes["receita_total"], marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(por_mes["mes"], por_mes["receita_total"], alpha=0.15, color="#2196F3")
    ax.set_title("Receita Total por Mês (2024)")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Receita Total (R$)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"], rotation=45)
    plt.tight_layout()
    caminho = os.path.join(output_dir, "vendas_por_mes.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico exportado: {caminho}")

      # --- Gráfico 2: Top 5 Produtos (barras horizontais) ---
    fig, ax = plt.subplots()
    top = metricas["top_produtos"]
    sns.barplot(data=top, y="produto", x="receita_total", ax=ax, palette="Blues_d", hue="produto")
    ax.set_title("Top 5 Produtos por Receita Total")
    ax.set_xlabel("Receita Total (R$)")
    ax.set_ylabel("Produto")
    for container in ax.containers: 
        ax.bar_label(container, fmt="R$ %.0f", padding=5)
    plt.tight_layout()
    caminho = os.path.join(output_dir, "top_produtos.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico exportado: {caminho}")

      # --- Gráfico 3: Distribuição de Receita por Região (boxplot) ---
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="regiao", y="receita_total", ax=ax, palette="Set2", hue = "regiao")
    ax.set_title("Distribuição de Receita por Transação – Por Região")
    ax.set_xlabel("Região")
    ax.set_ylabel("Receita por Venda (R$)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    caminho = os.path.join(output_dir, "distribuicao_regioes.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico exportado: {caminho}")

      # --- Gráfico 4 (Adicional 1): Histograma de Densidade das Vendas ---
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="receita_total", kde=True, ax=ax, color="#4CAF50", bins=30)
    ax.set_title("Frequência e Distribuição do Faturamento das Vendas")
    ax.set_xlabel("Valor da Receita por Item (R$)")
    ax.set_ylabel("Contagem de Transações")
    plt.tight_layout()
    caminho = os.path.join(output_dir, "distribuicao_faturamento_histograma.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico exportado: {caminho}")

      # --- Gráfico 5 (Adicional 2): Share de Faturamento por Categoria (Donut) ---
    fig, ax      = plt.subplots(figsize=(8, 8))
    cat_data = metricas["por_categoria"]
    
    ax.pie(cat_data["receita_total"], labels=cat_data["categoria"], autopct="%1.1f%%", 
           startangle = 90, colors = sns.color_palette("pastel"), pctdistance = 0.80,
           textprops  = {'fontsize': 12})
    
    circulo_central = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(circulo_central)
    
    ax.set_title("Participação (Share) de Faturamento por Categoria", fontsize=14)
    plt.tight_layout()
    caminho = os.path.join(output_dir, "share_faturamento_categoria.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico exportado: {caminho}")

    print("=== VISUALIZAÇÕES GERADAS COM SUCESSO ===\n")

# BLOCO PRINCIPAL
if __name__ == "__main__":
    path = "vendas.csv"
    
    try:
        # Carregamos os dados brutos do csv
        df_vendas = pd.read_csv(path)
        
        # Vamos executar a inspeção e leitura dos dados (RF02)
        inspecionar_dados(df_vendas)
        
        #RF03: Vamos executar a limpeza
        df_limpo, relatorio_limpeza = limpar_dados(df_vendas)
        
        #RF04: Vamos executar o enriquecimento
        df_enriquecido = criar_colunas_derivadas(df_limpo)

        #RF05: Vamos calcular as agregações estatísticas
        dicionario_metricas = calcular_metricas(df_enriquecido)

        #RF06: Vamos segmentar e classificar nossa carteira de clientes
        df_clientes_segmentados = segmentar_clientes(df_enriquecido)

        #RF07: Processamento estatístico matricial via NumPy
        estatisticas_np = calcular_estatisticas_numpy(df_enriquecido)

        #RF08: Vizualizações via seaborn e matplotlib
        gerar_visualizacoes(df_enriquecido, dicionario_metricas)
                        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{path}' não foi encontrado na raiz do projeto.")