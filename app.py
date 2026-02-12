# =============================================================
# app.py - Dashboard Crypto & Sentiment Analysis
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
st.set_page_config(
    page_title="Crypto & Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Chargement des données ---
@st.cache_data
def load_data():
    df = pd.read_csv("crypto_sentiment_prediction_dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['cryptocurrency', 'timestamp']).reset_index(drop=True)
    return df

df = load_data()

# =============================================================
# Sidebar et filtrage
# =============================================================

st.sidebar.title("🔧 Paramètres")

cryptos = df['cryptocurrency'].unique().tolist()
selected_crypto = st.sidebar.selectbox("Cryptomonnaie", cryptos)

date_min = df['timestamp'].min().date()
date_max = df['timestamp'].max().date()
date_range = st.sidebar.date_input("Plage de dates", [date_min, date_max],
                                    min_value=date_min, max_value=date_max)

page = st.sidebar.radio("Navigation", [
    "📈 Aperçu général",
    "🔗 Corrélations",
    "📉 Modélisation ARIMA vs ARIMAX",
    "⚡ Causalité de Granger",
    "📊 Comparaison multi-crypto"
])

# Filtrage
if len(date_range) == 2:
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    df_filtered = df[mask]
else:
    df_filtered = df.copy()

df_crypto = df_filtered[df_filtered['cryptocurrency'] == selected_crypto].copy()

# =============================================================
# PAGE 1 : Aperçu général
# =============================================================

if page == "📈 Aperçu général":
    st.title(f"📈 Aperçu général - {selected_crypto}")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prix actuel", f"${df_crypto['current_price_usd'].iloc[-1]:.2f}",
                f"{df_crypto['price_change_24h_percent'].iloc[-1]:.2f}%")
    col2.metric("Volume moyen 24h", f"${df_crypto['trading_volume_24h'].mean():,.0f}")
    col3.metric("Fear & Greed moyen", f"{df_crypto['fear_greed_index'].mean():.1f}")
    col4.metric("Observations", f"{len(df_crypto)}")

    st.markdown("---")

    # Prix + Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Prix (USD)", "Volume de trading"],
                        vertical_spacing=0.08, row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(x=df_crypto['timestamp'], y=df_crypto['current_price_usd'],
                             mode='lines', name='Prix', line=dict(color='#1f77b4')),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=df_crypto['timestamp'], y=df_crypto['trading_volume_24h'],
                         name='Volume', marker_color='rgba(31,119,180,0.3)'),
                  row=2, col=1)
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Sentiments
    st.subheader("Scores de sentiment")
    fig_sent = go.Figure()
    fig_sent.add_trace(go.Scatter(x=df_crypto['timestamp'],
                                   y=df_crypto['social_sentiment_score'],
                                   mode='lines', name='Social Sentiment',
                                   line=dict(color='#2ca02c')))
    fig_sent.add_trace(go.Scatter(x=df_crypto['timestamp'],
                                   y=df_crypto['news_sentiment_score'],
                                   mode='lines', name='News Sentiment',
                                   line=dict(color='#d62728')))
    fig_sent.update_layout(height=350)
    st.plotly_chart(fig_sent, use_container_width=True)

    # Fear & Greed + RSI
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Fear & Greed Index")
        fig_fg = px.line(df_crypto, x='timestamp', y='fear_greed_index',
                         color_discrete_sequence=['#ff7f0e'])
        fig_fg.add_hline(y=50, line_dash="dash", line_color="gray")
        fig_fg.update_layout(height=300)
        st.plotly_chart(fig_fg, use_container_width=True)

    with col_right:
        st.subheader("RSI")
        fig_rsi = px.line(df_crypto, x='timestamp', y='rsi_technical_indicator',
                          color_discrete_sequence=['#9467bd'])
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="Surachat")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="Survente")
        fig_rsi.update_layout(height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Stats descriptives
    st.subheader("Statistiques descriptives")
    st.dataframe(df_crypto.describe().round(3))

# =============================================================
# PAGE 2 : Corrélations
# =============================================================

elif page == "🔗 Corrélations":
    st.title(f"🔗 Analyse des corrélations - {selected_crypto}")

    text_signals = ['social_sentiment_score', 'news_sentiment_score',
                    'news_impact_score', 'social_mentions_count', 'fear_greed_index']
    numeric_cols = ['current_price_usd', 'price_change_24h_percent', 'trading_volume_24h',
                    'volatility_index', 'rsi_technical_indicator'] + text_signals

    # Matrice de corrélation
    st.subheader("Matrice de corrélation complète")
    corr = df_crypto[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1, aspect='auto')
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Corrélation signaux textuels vs prix
    st.subheader("Corrélation : Signaux textuels vs Variable cible")
    target = st.selectbox("Variable cible",
                          ['price_change_24h_percent', 'trading_volume_24h', 'current_price_usd'])

    corr_target = df_crypto[text_signals + [target]].corr()[target].drop(target)

    fig_bar = px.bar(x=corr_target.index, y=corr_target.values,
                     color=corr_target.values, color_continuous_scale='RdBu_r',
                     labels={'x': 'Signal', 'y': 'Corrélation'})
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # P-values
    st.subheader("Significativité des corrélations (p-values)")
    pvals = {}
    for signal in text_signals:
        corr_val, pval = stats.pearsonr(df_crypto[signal], df_crypto[target])
        pvals[signal] = {'Corrélation': round(corr_val, 4), 'P-value': round(pval, 4),
                         'Significatif (p<0.05)': '✓' if pval < 0.05 else '✗'}

    st.dataframe(pd.DataFrame(pvals).T)

    # Scatter plots
    st.subheader("Scatter plots")
    signal_choice = st.selectbox("Signal textuel", text_signals)

    fig_scatter = px.scatter(df_crypto, x=signal_choice, y=target,
                            color='fear_greed_index', 
                            color_continuous_scale='RdYlGn', 
                            opacity=0.6, 
                            trendline='ols',
                            hover_data=['timestamp'],
                            labels={'fear_greed_index': 'Fear & Greed Index'},
                            range_color=[0, 100])
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================
# PAGE 3 : Modélisation ARIMA vs ARIMAX
# =============================================================

elif page == "📉 Modélisation ARIMA vs ARIMAX":
    st.title(f"📉 Modélisation - {selected_crypto}")

    @st.cache_data
    def resample_data(df_in, crypto):
        subset = df_in[df_in['cryptocurrency'] == crypto].copy()
        subset = subset.set_index('timestamp').sort_index()
        numeric_only = subset.select_dtypes(include=[np.number])
        resampled = numeric_only.resample('6h').mean().interpolate(method='linear').dropna()
        return resampled

    data = resample_data(df_filtered, selected_crypto)

    st.write(f"Données resamplées (6h) : **{len(data)} observations**")

    # Paramètres
    col1, col2, col3 = st.columns(3)
    p = col1.slider("p (AR)", 0, 5, 2)
    d = col2.slider("d (différenciation)", 0, 2, 1)
    q = col3.slider("q (MA)", 0, 5, 2)

    train_pct = st.slider("% données d'entraînement", 50, 90, 80)

    exog_cols = ['social_sentiment_score', 'news_sentiment_score',
                 'news_impact_score', 'fear_greed_index', 'volatility_index']

    if st.button("Lancer la modélisation"):
        target = data['current_price_usd']
        exog = data[exog_cols]

        scaler = StandardScaler()
        exog_scaled = pd.DataFrame(scaler.fit_transform(exog),
                                    index=exog.index, columns=exog.columns)

        split = int(len(target) * train_pct / 100)
        train_y, test_y = target.iloc[:split], target.iloc[split:]
        train_exog, test_exog = exog_scaled.iloc[:split], exog_scaled.iloc[split:]

        # ARIMA
        with st.spinner("Entraînement ARIMA..."):
            try:
                model_arima = ARIMA(train_y, order=(p, d, q)).fit()
                pred_arima = model_arima.forecast(steps=len(test_y))
                mae_arima = mean_absolute_error(test_y, pred_arima)
                rmse_arima = np.sqrt(mean_squared_error(test_y, pred_arima))
            except Exception as e:
                st.error(f"Erreur ARIMA : {e}")
                mae_arima, rmse_arima, pred_arima = None, None, None

        # ARIMAX
        with st.spinner("Entraînement ARIMAX..."):
            try:
                model_arimax = ARIMA(train_y, exog=train_exog, order=(p, d, q)).fit()
                pred_arimax = model_arimax.forecast(steps=len(test_y), exog=test_exog)
                mae_arimax = mean_absolute_error(test_y, pred_arimax)
                rmse_arimax = np.sqrt(mean_squared_error(test_y, pred_arimax))
            except Exception as e:
                st.error(f"Erreur ARIMAX : {e}")
                mae_arimax, rmse_arimax, pred_arimax = None, None, None

        # Résultats
        st.subheader("Résultats")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ARIMA (prix seul)**")
            if mae_arima:
                st.write(f"MAE : {mae_arima:.4f}")
                st.write(f"RMSE : {rmse_arima:.4f}")
                st.write(f"AIC : {model_arima.aic:.2f}")

        with col2:
            st.markdown("**ARIMAX (prix + sentiment)**")
            if mae_arimax:
                st.write(f"MAE : {mae_arimax:.4f}")
                st.write(f"RMSE : {rmse_arimax:.4f}")
                st.write(f"AIC : {model_arimax.aic:.2f}")

        if mae_arima and mae_arimax:
            improvement = (mae_arima - mae_arimax) / mae_arima * 100
            if improvement > 0:
                st.success(f"ARIMAX améliore la MAE de {improvement:.2f}%")
            else:
                st.warning(f"ARIMA est meilleur (ARIMAX dégrade de {abs(improvement):.2f}%)")

        # Graphique des prédictions
        st.subheader("Prédictions vs Réel")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=train_y.index, y=train_y.values,
                                       mode='lines', name='Train',
                                       line=dict(color='gray')))
        fig_pred.add_trace(go.Scatter(x=test_y.index, y=test_y.values,
                                       mode='lines', name='Réel',
                                       line=dict(color='black', width=2)))
        if pred_arima is not None:
            fig_pred.add_trace(go.Scatter(x=test_y.index, y=pred_arima.values,
                                           mode='lines', name='ARIMA',
                                           line=dict(color='#1f77b4', dash='dash')))
        if pred_arimax is not None:
            fig_pred.add_trace(go.Scatter(x=test_y.index, y=pred_arimax.values,
                                           mode='lines', name='ARIMAX',
                                           line=dict(color='#d62728', dash='dash')))
        fig_pred.update_layout(height=500)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Résidus
        if pred_arimax is not None:
            st.subheader("Résidus ARIMAX")
            residuals = test_y.values - pred_arimax.values
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=test_y.index, y=residuals,
                                          mode='lines', name='Résidus',
                                          line=dict(color='steelblue')))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(height=300)
            st.plotly_chart(fig_res, use_container_width=True)

# =============================================================
# PAGE 4 : Causalité de Granger
# =============================================================

elif page == "⚡ Causalité de Granger":
    st.title(f"⚡ Causalité de Granger - {selected_crypto}")

    @st.cache_data
    def prepare_granger_data(df_in, crypto):
        subset = df_in[df_in['cryptocurrency'] == crypto].copy()
        subset = subset.set_index('timestamp').sort_index()
        numeric_only = subset.select_dtypes(include=[np.number])
        resampled = numeric_only.resample('6h').mean().interpolate(method='linear').dropna()

        for col in resampled.columns:
            _, pval, _, _, _, _ = adfuller(resampled[col].dropna())
            if pval >= 0.05:
                resampled[col] = resampled[col].diff()
        resampled = resampled.dropna()
        return resampled

    data_granger = prepare_granger_data(df_filtered, selected_crypto)

    cause_vars = ['social_sentiment_score', 'news_sentiment_score',
                  'news_impact_score', 'social_mentions_count', 'fear_greed_index']
    effect_vars = ['current_price_usd', 'price_change_24h_percent', 'trading_volume_24h']

    max_lag = st.slider("Nombre de lags maximum", 1, 8, 4)

    if st.button("Lancer les tests de Granger"):

        # Direction 1 : Sentiment → Prix
        st.subheader("Direction : Sentiment → Prix/Volume")
        results_fwd = {}
        for cause in cause_vars:
            for effect in effect_vars:
                pair_data = data_granger[[effect, cause]].dropna()
                if len(pair_data) > max_lag + 10:
                    try:
                        test = grangercausalitytests(pair_data, maxlag=max_lag, verbose=False)
                        best_pval = min([test[lag][0]['ssr_ftest'][1]
                                        for lag in range(1, max_lag + 1)])
                        best_lag = [lag for lag in range(1, max_lag + 1)
                                   if test[lag][0]['ssr_ftest'][1] == best_pval][0]
                        results_fwd[f"{cause} → {effect}"] = {
                            'P-value': round(best_pval, 4),
                            'Meilleur lag': best_lag,
                            'Significatif': '✓' if best_pval < 0.05 else '✗'
                        }
                    except:
                        pass

        df_fwd = pd.DataFrame(results_fwd).T
        st.dataframe(df_fwd)

        sig_fwd = sum(1 for v in results_fwd.values() if v['Significatif'] == '✓')
        st.write(f"**{sig_fwd} relations significatives** sur {len(results_fwd)} tests")

        # Heatmap
        pval_matrix = pd.DataFrame(index=cause_vars, columns=effect_vars, dtype=float)
        for pair, res in results_fwd.items():
            cause_name = pair.split(" → ")[0]
            effect_name = pair.split(" → ")[1]
            pval_matrix.loc[cause_name, effect_name] = res['P-value']

        fig_heat = px.imshow(pval_matrix.astype(float), text_auto='.3f',
                              color_continuous_scale='YlOrRd_r', zmin=0, zmax=0.1,
                              labels=dict(color="P-value"))
        fig_heat.update_layout(height=400, title="P-values : Sentiment → Prix/Volume")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # Direction 2 : Prix → Sentiment
        st.subheader("Direction : Prix/Volume → Sentiment")
        results_rev = {}
        for cause in effect_vars:
            for effect in cause_vars:
                pair_data = data_granger[[effect, cause]].dropna()
                if len(pair_data) > max_lag + 10:
                    try:
                        test = grangercausalitytests(pair_data, maxlag=max_lag, verbose=False)
                        best_pval = min([test[lag][0]['ssr_ftest'][1]
                                        for lag in range(1, max_lag + 1)])
                        best_lag = [lag for lag in range(1, max_lag + 1)
                                   if test[lag][0]['ssr_ftest'][1] == best_pval][0]
                        results_rev[f"{cause} → {effect}"] = {
                            'P-value': round(best_pval, 4),
                            'Meilleur lag': best_lag,
                            'Significatif': '✓' if best_pval < 0.05 else '✗'
                        }
                    except:
                        pass

        df_rev = pd.DataFrame(results_rev).T
        st.dataframe(df_rev)

        sig_rev = sum(1 for v in results_rev.values() if v['Significatif'] == '✓')
        st.write(f"**{sig_rev} relations significatives** sur {len(results_rev)} tests")

        # Comparaison
        st.subheader("Comparaison des directions")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='Sentiment → Prix', x=['Direction'],
                                   y=[sig_fwd], marker_color='steelblue'))
        fig_comp.add_trace(go.Bar(name='Prix → Sentiment', x=['Direction'],
                                   y=[sig_rev], marker_color='coral'))
        fig_comp.update_layout(barmode='group', height=300)
        st.plotly_chart(fig_comp, use_container_width=True)

# =============================================================
# PAGE 5 : Comparaison multi-crypto
# =============================================================

elif page == "📊 Comparaison multi-crypto":
    st.title("📊 Comparaison multi-crypto")

    selected_cryptos = st.multiselect("Sélectionner les cryptos", cryptos, default=cryptos[:5])

    if selected_cryptos:
        df_multi = df_filtered[df_filtered['cryptocurrency'].isin(selected_cryptos)]

        # Prix normalisé
        st.subheader("Prix normalisé (base 100)")
        fig_norm = go.Figure()
        for crypto in selected_cryptos:
            subset = df_multi[df_multi['cryptocurrency'] == crypto].copy()
            subset['price_norm'] = subset['current_price_usd'] / subset['current_price_usd'].iloc[0] * 100
            fig_norm.add_trace(go.Scatter(x=subset['timestamp'], y=subset['price_norm'],
                                           mode='lines', name=crypto))
        fig_norm.update_layout(height=500)
        st.plotly_chart(fig_norm, use_container_width=True)

        # Sentiments moyens
        st.subheader("Sentiment moyen par crypto")
        sentiment_avg = df_multi.groupby('cryptocurrency')[
            ['social_sentiment_score', 'news_sentiment_score']
        ].mean().round(3)

        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(name='Social Sentiment',
                                   x=sentiment_avg.index,
                                   y=sentiment_avg['social_sentiment_score'],
                                   marker_color='steelblue'))
        fig_sent.add_trace(go.Bar(name='News Sentiment',
                                   x=sentiment_avg.index,
                                   y=sentiment_avg['news_sentiment_score'],
                                   marker_color='coral'))
        fig_sent.update_layout(barmode='group', height=400)
        st.plotly_chart(fig_sent, use_container_width=True)

        # Volatilité et Fear & Greed
        st.subheader("Volatilité et Fear & Greed moyens")
        col1, col2 = st.columns(2)

        with col1:
            vol_avg = df_multi.groupby('cryptocurrency')['volatility_index'].mean().sort_values()
            fig_vol = px.bar(x=vol_avg.index, y=vol_avg.values,
                             color=vol_avg.values, color_continuous_scale='Reds',
                             labels={'x': 'Crypto', 'y': 'Volatilité moyenne'})
            fig_vol.update_layout(height=400, title="Volatilité moyenne")
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            fg_avg = df_multi.groupby('cryptocurrency')['fear_greed_index'].mean().sort_values()
            fig_fg = px.bar(x=fg_avg.index, y=fg_avg.values,
                            color=fg_avg.values, color_continuous_scale='RdYlGn',
                            labels={'x': 'Crypto', 'y': 'Fear & Greed moyen'})
            fig_fg.update_layout(height=400, title="Fear & Greed moyen")
            st.plotly_chart(fig_fg, use_container_width=True)

        # Corrélation par crypto
        st.subheader("Corrélation Sentiment → Variation de prix (par crypto)")
        text_signals = ['social_sentiment_score', 'news_sentiment_score',
                        'news_impact_score', 'social_mentions_count', 'fear_greed_index']

        corr_all = {}
        for crypto in selected_cryptos:
            subset = df_multi[df_multi['cryptocurrency'] == crypto]
            corr_all[crypto] = subset[text_signals + ['price_change_24h_percent']].corr()[
                'price_change_24h_percent'].drop('price_change_24h_percent')

        corr_all_df = pd.DataFrame(corr_all)
        fig_corr_all = px.imshow(corr_all_df, text_auto='.3f',
                                  color_continuous_scale='RdBu_r', zmin=-0.3, zmax=0.3)
        fig_corr_all.update_layout(height=400)
        st.plotly_chart(fig_corr_all, use_container_width=True)

        # Tableau récapitulatif
        st.subheader("Tableau récapitulatif")
        summary = df_multi.groupby('cryptocurrency').agg({
            'current_price_usd': ['mean', 'std'],
            'price_change_24h_percent': 'mean',
            'trading_volume_24h': 'mean',
            'volatility_index': 'mean',
            'fear_greed_index': 'mean',
            'social_sentiment_score': 'mean',
            'news_sentiment_score': 'mean'
        }).round(3)
        summary.columns = ['Prix moyen', 'Prix std', 'Variation moy.', 'Volume moyen',
                          'Volatilité moy.', 'Fear&Greed moy.', 'Sent. Social moy.',
                          'Sent. News moy.']
        st.dataframe(summary)
