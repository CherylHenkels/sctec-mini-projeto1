import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re

# ==============================================================================
# RF02 – LEITURA E INSPEÇÃO DOS DADOS
# ==============================================================================

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

# ==============================================================================
# RF03 – LIMPEZA DOS DADOS
# ==============================================================================

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

# ==============================================================================
# RF04 – CRIAR COLUNAS DERIVADAS
# ==============================================================================

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

# ==============================================================================
# RF05 – CALCULAR MÉTRICAS AGREGADAS
# ==============================================================================

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

# ==============================================================================
# RF06 – SEGMENTAR CLIENTES
# ==============================================================================

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

# ==============================================================================
# RF07 – CALCULAR ESTATÍSTICAS
# ==============================================================================

def calcular_estatisticas_numpy(df):
    """RF07 – Calcular Estatísticas Avançadas com NumPy."""
    print("=== ESTATÍSTICAS MATRICIAIS COM NUMPY ===")

    #Conversão da coluna do DataFrame para array NumPy bruto
    receitas = df["receita_total"].to_numpy()

    # Uso de múltiplas funções estatísticas nativas do NumPy
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

# ==============================================================================
# RF08 – VISUALIZAÇÕES
# ==============================================================================

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

# ==============================================================================
# RF13 – USAR EXPRESSÕES REGULARES PARA LIMPEZA DE DADOS
# ==============================================================================

def limpar_strings_com_regex(df):
    """RF13 – Usa expressões regulares para limpeza de colunas de texto."""
    # Criamos uma cópia para trabalhar com segurança
    df_foco = df.copy()

    # 1. Remover caracteres não alfanuméricos do nome do cliente (exceto underline e espaço)
    df_foco["cliente_limpo"] = df_foco["cliente"].apply(
        lambda s: re.sub(r"[^a-zA-Z0-9_ ]", "", str(s)).strip()
    )

    # 2. Identificar registros com padrão de ID inválido (deve ser "Cliente_XXX")
    padrao_cliente = re.compile(r"^Cliente_\d{3}$")
    df_foco["cliente_valido"] = df_foco["cliente_limpo"].apply(
        lambda s: bool(padrao_cliente.match(str(s)))
    )

    n_invalidos = (~df_foco["cliente_valido"]).sum()
    print(f"=== LIMPEZA E VALIDAÇÃO COM REGEX (RF13) ===")
    print(f"  Clientes com formato inválido encontrados: {n_invalidos}")
    print(f"  Amostra de clientes limpos: {df_foco['cliente_limpo'].head(3).tolist()}")
    print("============================================\n")

    # Substitui a coluna original pela limpa e remove a coluna de validação para não sujar o DF
    df_foco["cliente"] = df_foco["cliente_limpo"]
    df_foco = df_foco.drop(columns=["cliente_limpo", "cliente_valido"])

    return df_foco

# ==============================================================================
# RF09 - CLASSE BASE. CAMADA DE ORIENTAÇÃO A OBJETOS
# ==============================================================================

class AnalisadorDeVendas:
    """RF09 – Classe responsável por encapsular o pipeline analítico de vendas.
    Mantém o estado dos DataFrames e os resultados intermediários na memória RAM.
    """

    def __init__(self, caminho_arquivo):
        """Inicializa o analisador com o caminho do arquivo de dados."""
        self.caminho_arquivo = caminho_arquivo
        self.df_bruto = None
        self.df_limpo = None
        self.metricas = {}
        self.clientes = None       # Guardará o retorno do RF06
        self.stats_numpy = {}      # Guardará o retorno do RF07
        self.relatorio_limpeza = {}

    def carregar(self):
        """Lê o arquivo CSV e armazena o DataFrame bruto."""
        self.df_bruto = pd.read_csv(self.caminho_arquivo)
        print(f"[POO] Dados brutos carregados de: {self.caminho_arquivo}")
        print(f"      Registros encontrados: {len(self.df_bruto)}")
        return self

    # RF02: Inspecionar e Descrever os Dados
    def inspecionar(self):
        """Chama a função de inspeção inicial do dataset."""
        inspecionar_dados(self.df_bruto)
        return self

    # RF03: Limpar e Tratar os Dados
    def limpar(self):
        """Limpa os dados chamando a lógica estruturada do RF03 e o Regex do RF13."""
        self.df_limpo, self.relatorio_limpeza = limpar_dados(self.df_bruto.copy())

        self.df_limpo = limpar_strings_com_regex(self.df_limpo)
        return self

    # RF04: Criar Colunas Derivadas
    def transformar(self):
        """Aplica transformações e cria as colunas derivadas do RF04."""
        self.df_limpo = criar_colunas_derivadas(self.df_limpo)

        print("[Lambda] Calculando coluna de desconto condicional dinâmico...")
        self.df_limpo["desconto"] = self.df_limpo["receita_total"].apply(
            lambda x: 0.10 if x > 10000 else 0.05
        )
        return self

    # RF05, RF06 e RF07: Agregações, Segmentação de Clientes e Estatísticas NumPy
    def analisar(self):
        """Calcula de forma centralizada as agregações, segmentações e métricas NumPy."""
        self.metricas = calcular_metricas(self.df_limpo)             # RF05
        self.clientes = segmentar_clientes(self.df_limpo)            # RF06
        self.stats_numpy = calcular_estatisticas_numpy(self.df_limpo) # RF07
        return self

    # RF08: Visualizações
    def visualizar(self):
        """Dispara a geração de relatórios gráficos do RF08."""
        gerar_visualizacoes(self.df_limpo, self.metricas)
        return self
    
    # RF11: Usar Funções Lambda e Funções de Ordem Superior
    def processar_coluna(self, coluna, funcao_transformacao):
        """Aplica uma função de transformação a uma coluna do DataFrame.
        Demonstra o uso de funções como argumentos (higher-order function / callback).
        """
        if self.df_limpo is not None and coluna in self.df_limpo.columns:
            self.df_limpo[f"{coluna}_transformado"] = self.df_limpo[coluna].apply(funcao_transformacao)
            print(f"  [Callback] Coluna '{coluna}_transformado' criada com sucesso via injeção funcional.")
        return self

    def resumo(self):
        """Exibe um resumo executivo consolidado no console."""
        print("\n" + "="*50)
        print("         RESUMO EXECUTIVO – SALESINSIGHT PY")
        print("="*50)
        print(f"  Arquivo analisado:      {self.caminho_arquivo}")
        print(f"  Registros brutos:       {self.relatorio_limpeza.get('registros_iniciais', 'N/A')}")
        print(f"  Registros limpos:       {self.relatorio_limpeza.get('registros_finais', 'N/A')}")
        receita = self.df_limpo["receita_total"].sum() if self.df_limpo is not None else 0
        print(f"  Receita total anual:    R$ {receita:,.2f}")
        if self.clientes is not None and not self.clientes.empty:
            top = self.clientes.iloc[0]
            print(f"  Cliente TOP 1:          {top['cliente']} (R$ {top['total_gasto']:,.2f})")
        print("="*50 + "\n")
        return self
    
    # RF12: LER E ESCREVER ARQUIVOS (CSV E JSON)
    def exportar_resultados(self):
        """Exporta resultados intermediários e métricas do pipeline em formatos CSV e JSON.
        Demonstra escrita em lote e leitura de confirmação ativa.
        """
        if not self.metricas or self.clientes is None or not self.stats_numpy:
            print("[AVISO] Rode .analisar() antes de tentar exportar os resultados.")
            return self

        os.makedirs("outputs", exist_ok=True)

        # Exportar CSV com métricas por mês
        caminho_csv = "outputs/metricas_por_mes.csv"
        self.metricas["por_mes"].to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        print(f"  [Escrita] CSV exportado com sucesso: {caminho_csv}")

        # Exportar segmentação de clientes em CSV
        caminho_clientes = "outputs/segmentacao_clientes.csv"
        self.clientes.to_csv(caminho_clientes, index=False, encoding="utf-8-sig")
        print(f"  [Escrita] CSV exportado com sucesso: {caminho_clientes}")

        # Exportar estatísticas gerais computadas via NumPy em JSON
        caminho_json = "outputs/estatisticas_gerais.json"
        # O list comprehension blinda contra floats brutos do numpy incompatíveis com JSON nativo
        stats_serializaveis = {k: round(float(v), 2) for k, v in self.stats_numpy.items()}
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(stats_serializaveis, f, indent=4, ensure_ascii=False)
        print(f"  [Escrita] JSON exportado com sucesso: {caminho_json}")

        # Ler e exibir o JSON exportado para confirmar de forma ativa
        print("\n  === [Leitura] RE-LENDO ARQUIVO JSON GERADO PARA CONFIRMAÇÃO ===")
        with open(caminho_json, "r", encoding="utf-8") as f:
            dados_lidos = json.load(f)
        print(f"  Conteúdo lido com sucesso:\n{json.dumps(dados_lidos, indent=4)}")
        print("  ===============================================================\n")
        
        return self

# ==============================================================================
# RF10 – USAR HERANÇA (CLASSE DERIVADA)
# ==============================================================================

class AnalisadorComProjecao(AnalisadorDeVendas):
    """Extensão do AnalisadorDeVendas com funcionalidades de projeção simples.
    Herda todos os métodos da classe pai e adiciona projeção de tendência.
    """

    def __init__(self, caminho_arquivo, meses_projecao=3):
        super().__init__(caminho_arquivo)
        self.meses_projecao = meses_projecao
        self.projecoes = []

    def projetar_tendencia(self):
        """Projeta a receita dos próximos meses com base na média móvel dos últimos 3 meses.
        Método simples sem machine learning – baseado em médias.
        """
        if not self.metricas or "por_mes" not in self.metricas:
            print("[AVISO] Rode .analisar() antes de projetar.")
            return self

        por_mes = self.metricas["por_mes"].sort_values("mes")
        receitas_historicas = por_mes["receita_total"].to_numpy()

        # Média móvel dos últimos 3 meses como base da projeção
        ultimos_3 = receitas_historicas[-3:]
        media_movel = np.mean(ultimos_3)
        tendencia = np.std(ultimos_3) * 0.1  # fator de crescimento simples

        ultimo_mes = int(por_mes["mes"].max())

        print("\n=== PROJEÇÃO DE TENDÊNCIA (Média Móvel Simples) ===")
        print(f"  Base: média dos últimos 3 meses = R$ {media_movel:,.2f}")
        self.projecoes = []

        for i in range(1, self.meses_projecao + 1):
            mes_projetado = (ultimo_mes + i - 1) % 12 + 1
            receita_projetada = media_movel + (tendencia * i)
            self.projecoes.append({"mes": mes_projetado, "receita_projetada": round(receita_projetada, 2)})
            print(f"  Mês {mes_projetado:02d} (projeção): R$ {receita_projetada:,.2f}")

        return self

    def exibir_projecao_detalhada(self):
        """Exibe o detalhamento das projeções calculadas."""
        if not self.projecoes:
            print("[AVISO] Nenhuma projeção disponível. Rode .projetar_tendencia() primeiro.")
            return self
        print("\n=== DETALHAMENTO DAS PROJEÇÕES ===")
        for p in self.projecoes:
            print(f"  Mês {p['mes']:02d}: R$ {p['receita_projetada']:,.2f}")
        print("==================================\n")
        return self

# BLOCO PRINCIPAL
def main():
    path = "vendas.csv"
    
    print("\n" + "="*60)
    print("   SALESINSIGHT PY – Pipeline de Análise de Dados de Vendas")
    print("="*60 + "\n")
    
    try:
        print("Iniciando pipeline por meio da Classe AnalisadorComProjecao...\n")
        
        # Instanciamos a classe FILHA passando o arquivo de dados
        analisador = AnalisadorComProjecao(path, meses_projecao=3)
        
        # Executamos o pipeline base até a visualização gráfica
        (analisador
         .carregar()       # Carregamos o arquivo
         .inspecionar()    # Executa o RF02 (Inspeção)
         .limpar()         # Executa o RF03 (Limpeza)
         .transformar()    # Executa o RF04 (Colunas derivadas + Lambda Interno)
         .analisar()       # Executa o RF05, RF06 (Clientes) e RF07 (NumPy)
         .visualizar())    # Executa o RF08 (Gráficos)
        
        # ----------------------------------------------------------------------
        # RF11: USO DA FUNÇÃO DE ORDEM SUPERIOR COM CALLBACKS (LAMBDAS)
        # ----------------------------------------------------------------------
        print("\n--- APLICANDO CALLBACKS DINÂMICOS (RF11) ---")
        # Callback 1: Normalização de escala da receita por item (Expressa em kR$)
        analisador.processar_coluna("receita_total", lambda x: round(x / 1000, 2))
        
        # Callback 2: Classificação qualitativa de volumes por lote vendido
        analisador.processar_coluna("quantidade", lambda x: "Alto" if x > 5 else "Baixo")
        print("--------------------------------------------------\n")
        # ----------------------------------------------------------------------
        
        # Executamos as projeções (RF10) e o fechamento com o resumo executivo
        (analisador
         .projetar_tendencia()            # Executa o RF10 (Método da classe filha)
         .exibir_projecao_detalhada()     # Executa o RF10 (Método da classe filha)
         .exportar_resultados()           # RF12
         .resumo())                       # Mostra o painel final consolidado
        
        print("[Sucesso] O pipeline orientado a objetos com Herança e Callbacks rodou perfeitamente!")
                        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{path}' não foi encontrado na raiz do projeto.")

if __name__ == "__main__":
    main()        