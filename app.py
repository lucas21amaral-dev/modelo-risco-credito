"""
=============================================================================
  APP STREAMLIT — Análise de Risco de Crédito
  Coloque este arquivo na mesma pasta que:
    - pipeline_credito.joblib
    - config_deploy.joblib
  Rode com: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# CLASSES DO PIPELINE (necessárias para desserializar o .joblib)
# Devem ser idênticas às usadas no treino.
# ============================================================
COLUNA_ALVO = 'inadimplente_2anos'


def calcular_parametros_treino(df_treino):
    return {
        'mediana_renda': df_treino['renda_mensal'].median(),
        'mediana_dependentes': df_treino['dependentes'].median(),
        'corte_atrasos': 20,
        'codigos_atraso_especiais': [96, 98],
        'corte_renda': 50000,
    }


def preparar_dados(df, parametros):
    df = df.copy()
    df['flag_renda_ausente'] = df['renda_mensal'].isna().astype(int)
    df['renda_mensal'] = df['renda_mensal'].fillna(parametros['mediana_renda'])
    df['flag_dependentes_ausente'] = df['dependentes'].isna().astype(int)
    df['dependentes'] = df['dependentes'].fillna(parametros['mediana_dependentes'])
    colunas_atraso = ['atrasos_30_59_dias', 'atrasos_60_89_dias', 'atrasos_90_mais_dias']
    df['flag_atraso_codigo_especial'] = df['atrasos_30_59_dias'].isin(
        parametros['codigos_atraso_especiais']
    ).astype(int)
    for col in colunas_atraso:
        df[col] = df[col].clip(upper=parametros['corte_atrasos'])
    df['renda_mensal'] = df['renda_mensal'].clip(upper=parametros['corte_renda'])
    df['renda_por_dependente'] = df['renda_mensal'] / (df['dependentes'] + 1)
    df['sobra_caixa'] = df['renda_mensal'] * (1 - df['razao_divida'])
    return df


class PreparadorCredito(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.parametros_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        self.parametros_ = calcular_parametros_treino(X)
        amostra = preparar_dados(X.head(5), self.parametros_)
        if COLUNA_ALVO in amostra.columns:
            amostra = amostra.drop(columns=[COLUNA_ALVO])
        self.feature_names_out_ = amostra.columns.tolist()
        return self

    def transform(self, X):
        df = preparar_dados(X, self.parametros_)
        if COLUNA_ALVO in df.columns:
            df = df.drop(columns=[COLUNA_ALVO])
        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class ImputadorSeguranca(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer_ = SimpleImputer(strategy='median')
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_out_ = X.columns.tolist()
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        colunas = X.columns.tolist() if isinstance(X, pd.DataFrame) else self.feature_names_out_
        resultado = self.imputer_.transform(X)
        return pd.DataFrame(resultado, columns=colunas,
                            index=X.index if isinstance(X, pd.DataFrame) else None)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# ============================================================
# TRADUÇÃO DOS NOMES DAS FEATURES PARA EXIBIÇÃO
# ============================================================
NOMES_DISPLAY = {
    'idade': 'Idade',
    'renda_mensal': 'Renda Mensal',
    'dependentes': 'Dependentes',
    'uso_limite_rotativo': 'Uso do Limite Rotativo',
    'razao_divida': 'Razão Dívida/Renda',
    'linhas_credito_abertas': 'Linhas de Crédito Abertas',
    'financiamentos_imobiliarios': 'Financiamentos Imobiliários',
    'atrasos_30_59_dias': 'Atrasos 30-59 dias',
    'atrasos_60_89_dias': 'Atrasos 60-89 dias',
    'atrasos_90_mais_dias': 'Atrasos 90+ dias',
    'flag_renda_ausente': 'Renda Não Informada',
    'flag_dependentes_ausente': 'Dependentes Não Informado',
    'flag_atraso_codigo_especial': 'Código Especial de Atraso',
    'renda_por_dependente': 'Renda por Dependente',
    'sobra_caixa': 'Sobra de Caixa',
}


# ============================================================
# CARREGAR MODELO E CONFIG (cache para não recarregar)
# ============================================================
@st.cache_resource
def carregar_modelo():
    pipeline = joblib.load('pipeline_credito.joblib')
    config = joblib.load('config_deploy.joblib')
    modelo_xgb = pipeline.named_steps['modelo']
    explainer = shap.TreeExplainer(modelo_xgb)
    return pipeline, config, explainer


# ============================================================
# INTERFACE
# ============================================================
st.set_page_config(
    page_title='Análise de Risco de Crédito',
    page_icon='🏦',
    layout='centered',
)

# CSS customizado
st.markdown("""
<style>
    .resultado-aprovar {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        margin: 16px 0;
    }
    .resultado-negar {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
        margin: 16px 0;
    }
    .resultado-titulo {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .resultado-prob {
        font-size: 18px;
        margin-bottom: 4px;
    }
    .metrica-box {
        background-color: #f0f2f6;
        padding: 12px 16px;
        border-radius: 8px;
        text-align: center;
    }
    .shap-positivo { color: #dc3545; font-weight: 600; }
    .shap-negativo { color: #28a745; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title('🏦 Análise de Risco de Crédito')
st.markdown('Preencha as informações do cliente para obter a análise de risco.')

pipeline, config, explainer = carregar_modelo()

st.divider()

# ============================================================
# FORMULÁRIO
# ============================================================
with st.form('formulario_cliente'):

    st.subheader('📋 Dados do Cliente')

    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input(
            'Idade', min_value=18, max_value=120,
            value=35, step=1,
        )
        renda_informada = st.checkbox('Renda informada?', value=True)
        if renda_informada:
            renda_mensal = st.number_input(
                'Renda Mensal (R$)', min_value=0.0, max_value=500000.0,
                value=5000.0, step=100.0, format='%.2f',
            )
        else:
            renda_mensal = None
            st.info('Renda será imputada pela mediana do treino.')

        dependentes_informado = st.checkbox('Dependentes informado?', value=True)
        if dependentes_informado:
            dependentes = st.number_input(
                'Número de Dependentes', min_value=0, max_value=20,
                value=0, step=1,
            )
        else:
            dependentes = None
            st.info('Dependentes serão imputados pela mediana do treino.')

    with col2:
        uso_limite_rotativo = st.number_input(
            'Uso do Limite Rotativo (%)',
            min_value=0.0, max_value=200.0,
            value=30.0, step=1.0,
            help='Quanto do limite de crédito rotativo está sendo usado. Ex: 50 = 50%.',
        )
        razao_divida = st.number_input(
            'Razão Dívida/Renda',
            min_value=0.0, max_value=2.0,
            value=0.3, step=0.01, format='%.2f',
            help='Proporção da dívida em relação à renda. Ex: 0.3 = 30%.',
        )
        linhas_credito = st.number_input(
            'Linhas de Crédito Abertas',
            min_value=0, max_value=60, value=5, step=1,
        )
        financiamentos = st.number_input(
            'Financiamentos Imobiliários',
            min_value=0, max_value=60, value=0, step=1,
        )

    st.subheader('⚠️ Histórico de Atrasos (últimos 2 anos)')

    col3, col4, col5 = st.columns(3)

    with col3:
        atrasos_30_59 = st.number_input(
            'Atrasos 30-59 dias', min_value=0, max_value=20,
            value=0, step=1,
        )
    with col4:
        atrasos_60_89 = st.number_input(
            'Atrasos 60-89 dias', min_value=0, max_value=20,
            value=0, step=1,
        )
    with col5:
        atrasos_90_mais = st.number_input(
            'Atrasos 90+ dias', min_value=0, max_value=20,
            value=0, step=1,
        )

    st.divider()
    enviado = st.form_submit_button(
        '🔍 Analisar Risco',
        use_container_width=True,
        type='primary',
    )

# ============================================================
# PREDIÇÃO E RESULTADO
# ============================================================
if enviado:

    # Montar DataFrame do cliente (formato cru, como o pipeline espera)
    dados_cliente = pd.DataFrame([{
        'inadimplente_2anos': 0,  # placeholder, removido pelo pipeline
        'idade': int(idade),
        'renda_mensal': float(renda_mensal) if renda_mensal is not None else np.nan,
        'dependentes': float(dependentes) if dependentes is not None else np.nan,
        'uso_limite_rotativo': uso_limite_rotativo / 100.0,  # converter % para proporção
        'razao_divida': razao_divida,
        'linhas_credito_abertas': int(linhas_credito),
        'financiamentos_imobiliarios': int(financiamentos),
        'atrasos_30_59_dias': int(atrasos_30_59),
        'atrasos_60_89_dias': int(atrasos_60_89),
        'atrasos_90_mais_dias': int(atrasos_90_mais),
    }])

    # Predição
    probabilidade = pipeline.predict_proba(dados_cliente)[:, 1][0]
    limiar = config['limiar']
    decisao = 'NEGAR' if probabilidade >= limiar else 'APROVAR'

    # ============================================================
    # EXIBIR RESULTADO
    # ============================================================
    st.divider()
    st.subheader('📊 Resultado da Análise')

    if decisao == 'APROVAR':
        st.markdown(f"""
        <div class="resultado-aprovar">
            <div class="resultado-titulo">✅ CRÉDITO APROVADO</div>
            <div class="resultado-prob">
                Probabilidade de inadimplência: <b>{probabilidade*100:.1f}%</b>
            </div>
            <div>Limiar de decisão: {limiar*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="resultado-negar">
            <div class="resultado-titulo">❌ CRÉDITO NEGADO</div>
            <div class="resultado-prob">
                Probabilidade de inadimplência: <b>{probabilidade*100:.1f}%</b>
            </div>
            <div>Limiar de decisão: {limiar*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Métricas lado a lado
    m1, m2, m3 = st.columns(3)
    m1.metric('Probabilidade', f'{probabilidade*100:.1f}%')
    m2.metric('Limiar', f'{limiar*100:.0f}%')
    m3.metric('Decisão', decisao)

    # ============================================================
    # ANÁLISE SHAP — TOP 5 FEATURES
    # ============================================================
    st.divider()
    st.subheader('🔍 Principais fatores da decisão (SHAP)')

    # Preparar dados para SHAP
    X_preparado = pipeline[:-1].transform(dados_cliente)
    nomes_features = config['nomes_features']

    shap_values = explainer.shap_values(X_preparado)

    # Se retornar matriz, pegar primeira linha
    if len(shap_values.shape) > 1:
        shap_cliente = shap_values[0]
    else:
        shap_cliente = shap_values

    # Top 5 por valor absoluto
    indices_top5 = np.argsort(np.abs(shap_cliente))[::-1][:5]

    # Tabela
    linhas_tabela = []
    for idx in indices_top5:
        nome_feat = nomes_features[idx]
        nome_display = NOMES_DISPLAY.get(nome_feat, nome_feat)
        valor_feat = X_preparado.iloc[0, idx]
        valor_shap = shap_cliente[idx]

        if valor_shap > 0:
            direcao = '↑ Aumenta risco'
        else:
            direcao = '↓ Reduz risco'

        linhas_tabela.append({
            'Feature': nome_display,
            'Valor do Cliente': f'{valor_feat:.2f}',
            'Impacto (SHAP)': f'{valor_shap:+.4f}',
            'Direção': direcao,
        })

    df_shap = pd.DataFrame(linhas_tabela)
    df_shap.index = range(1, len(df_shap) + 1)
    df_shap.index.name = '#'

    st.table(df_shap)

    # Gráfico de barras SHAP (top 5)
    fig, ax = plt.subplots(figsize=(8, 3.5))

    nomes_top5 = [NOMES_DISPLAY.get(nomes_features[i], nomes_features[i]) for i in indices_top5]
    valores_top5 = [shap_cliente[i] for i in indices_top5]
    cores = ['#dc3545' if v > 0 else '#28a745' for v in valores_top5]

    # Inverter para o mais importante ficar no topo
    nomes_top5 = nomes_top5[::-1]
    valores_top5 = valores_top5[::-1]
    cores = cores[::-1]

    ax.barh(nomes_top5, valores_top5, color=cores, height=0.6, edgecolor='none')
    ax.set_xlabel('Impacto SHAP (→ aumenta risco | ← reduz risco)', fontsize=10)
    ax.axvline(x=0, color='#333', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Explicação textual
    st.divider()
    st.subheader('📝 Explicação para o cliente')

    top1_nome = NOMES_DISPLAY.get(nomes_features[indices_top5[0]], nomes_features[indices_top5[0]])
    top1_shap = shap_cliente[indices_top5[0]]

    if decisao == 'NEGAR':
        fatores_risco = [
            NOMES_DISPLAY.get(nomes_features[i], nomes_features[i])
            for i in indices_top5 if shap_cliente[i] > 0
        ]
        if fatores_risco:
            texto_fatores = ', '.join(fatores_risco[:3])
            st.warning(
                f'O crédito foi negado porque a probabilidade de inadimplência '
                f'({probabilidade*100:.1f}%) superou o limiar de {limiar*100:.0f}%. '
                f'Os principais fatores de risco foram: **{texto_fatores}**.'
            )
        else:
            st.warning(
                f'O crédito foi negado porque a probabilidade de inadimplência '
                f'({probabilidade*100:.1f}%) superou o limiar de {limiar*100:.0f}%.'
            )
    else:
        fatores_positivos = [
            NOMES_DISPLAY.get(nomes_features[i], nomes_features[i])
            for i in indices_top5 if shap_cliente[i] < 0
        ]
        if fatores_positivos:
            texto_fatores = ', '.join(fatores_positivos[:3])
            st.success(
                f'O crédito foi aprovado com probabilidade de inadimplência de '
                f'{probabilidade*100:.1f}% (abaixo do limiar de {limiar*100:.0f}%). '
                f'Fatores favoráveis: **{texto_fatores}**.'
            )
        else:
            st.success(
                f'O crédito foi aprovado com probabilidade de inadimplência de '
                f'{probabilidade*100:.1f}% (abaixo do limiar de {limiar*100:.0f}%).'
            )

# ============================================================
# RODAPÉ
# ============================================================
st.divider()
st.caption(
    f'Modelo: {config["nome_modelo"]} | '
    f'Limiar: {config["limiar"]} | '
    f'Custo FN: R$ {config["custo_fn"]:,} | '
    f'Custo FP: R$ {config["custo_fp"]:,}'
)
