import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from features import prepare_dataset
from preprocessing import preprocess
from models import train_models_grid
from visualization import plot_heatmap, plot_bar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(
    page_title="SmartInvest AI – Financial Analytics",
    page_icon="💼",
    layout="wide",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #0f172a; }
.stApp { background: linear-gradient(180deg, #fdfdfd 0%, #f3f4f6 100%); }
h1, h2, h3, h4, h5 { color: #0f172a; font-weight: 700; }
p, div, span { color: #1e293b; }

.metric-card {
    background-color: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    padding: 20px; text-align: center; transition: 0.3s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }

div.stButton > button {
    background-color: #1a73e8; color: #ffffff; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 1.2rem; transition: 0.3s;
}
div.stButton > button:hover { background-color: #155bb5; transform: scale(1.03); }

div.stSelectbox > div > div > select {
    color: #ffffff ; background-color: #1a73e8 ; border-radius: 6px; padding: 4px;
}
div.stSelectbox div[role="listbox"] div {
    color: #1e293b ; background-color: #f3f4f6 ;
}

.nav-container { display: flex; justify-content: center; background-color: #ffffff; border-radius: 12px; padding: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem; }
.nav-item { margin: 0 25px; font-weight: 600; color: #0f172a; text-decoration: none; cursor: pointer; transition: color 0.2s ease; }
.nav-item:hover { color: #1a73e8; }
.active-nav { color: #1a73e8; border-bottom: 2px solid #1a73e8; }

.fin-highlight { color: #0a8754; font-weight: 700; }
.fin-risk { color: #c0392b; font-weight: 700; }

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;">
<h1 style='text-align:left;'>💹 SmartInvest AI — Financial Analytics Dashboard</h1>
</div>
<p style='text-align:left; color:#334155; font-size:16px;'>
Analyse de marché et intelligence financière basées sur le Machine Learning.
</p>
""", unsafe_allow_html=True)

nav_options = ["📊 Aperçu", "🏦 Marché", "🤖 Modélisation ML", "💰 Prédictions"]
col_nav = st.columns(len(nav_options))
for i, opt in enumerate(nav_options):
    if col_nav[i].button(opt):
        st.session_state["page"] = opt
if "page" not in st.session_state:
    st.session_state["page"] = "📊 Aperçu"
page = st.session_state["page"]

st.markdown("<br>", unsafe_allow_html=True)

df_feat = prepare_dataset()
tickers = df_feat['Ticker'].unique()
ticker_selected = st.selectbox("📊 Sélectionner un actif :", tickers, index=0)
df_ticker = df_feat[df_feat['Ticker'] == ticker_selected]

if page == "📊 Aperçu":
    st.markdown("## 📊 Aperçu des données financières")
    st.subheader("Statistiques descriptives")
    st.dataframe(df_ticker.describe(), width='stretch')

 
    features = ['Return', 'SMA_7', 'EMA_7', 'Volatility_7']
    feature_selected = st.selectbox("Choisir une feature pour visualisation :", features)

    fig_hist = px.histogram(df_ticker, x=feature_selected, nbins=30, title=f"Histogramme de {feature_selected}")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(df_ticker, y=feature_selected, points="all", title=f"Box plot de {feature_selected}")
    st.plotly_chart(fig_box, use_container_width=True)

    if "Date" in df_ticker.columns:
        fig_line = px.line(df_ticker, x="Date", y=feature_selected, title=f"Évolution de {feature_selected} dans le temps")
        st.plotly_chart(fig_line, use_container_width=True)

elif page == "🏦 Marché":
    st.markdown(f"## 📊 Marché — {ticker_selected}")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown("#### Données récentes")
        st.dataframe(df_ticker.tail(10), width='stretch')
    with col2:
        st.markdown("#### Indicateurs clés")
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Prix actuel", f"{df_ticker['Adj Close'].iloc[-1]:.2f} $")
        st.metric("Rendement 7j", f"{(df_ticker['Return'].iloc[-7:].mean()*100):.2f}%")
        st.metric("Volatilité 7j", f"{df_ticker['Volatility_7'].iloc[-1]:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("#### Statistiques générales")
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Prix max", f"{df_ticker['Adj Close'].max():.2f} $")
        st.metric("Prix min", f"{df_ticker['Adj Close'].min():.2f} $")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🤖 Modélisation ML":
    st.markdown(f"## 🤖 Modélisation ML — {ticker_selected}")

    X_train, X_test, y_train, y_test, scaler = preprocess(df_ticker)
    st.session_state["scaler"] = scaler

    st.markdown("### ⚙️ Entraînement des modèles")
    if st.button("🚀 Entraîner tous les modèles"):
        with st.spinner("Entraînement en cours…"):
            best_models, results = train_models_grid(X_train, y_train)
        st.success("✅ Entraînement terminé !")
        st.session_state["best_models"] = best_models
        st.session_state["results"] = results

    st.markdown("### 📊 Évaluation complète des modèles")
    if "best_models" in st.session_state:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        import seaborn as sns

        fig_roc, ax_roc = plt.subplots(figsize=(8,6))

        for name, model in st.session_state["best_models"].items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

        ax_roc.plot([0,1], [0,1], linestyle='--', color='gray')
        ax_roc.set_title("ROC Curve comparatives")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

        model_choice = st.selectbox("Choisir un modèle pour afficher sa matrice de confusion :", list(st.session_state["best_models"].keys()))
        model = st.session_state["best_models"][model_choice]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        st.markdown(f"### 📌 Évaluation du modèle : {model_choice}")
        st.markdown(f"- Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        st.markdown(f"- F1-score : {f1_score(y_test, y_pred):.4f}")
        st.markdown(f"- Precision : {precision_score(y_test, y_pred):.4f}")
        st.markdown(f"- Recall : {recall_score(y_test, y_pred):.4f}")
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            st.markdown(f"- ROC-AUC : {auc:.4f}")

            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            sensitivity = tpr[optimal_idx]
            specificity = 1 - fpr[optimal_idx]

            st.markdown("### 🔍 Interprétation de la ROC")
            st.markdown(f"""
        - **AUC** = {auc:.2f} — {"Très performant ✅" if auc>0.85 else "Modérément performant ⚠️" if auc>0.7 else "Faible ❌"}
        - **Seuil optimal** = {optimal_threshold:.2f}  
        Le seuil à partir duquel le modèle classifie un événement comme positif.
        - **Sensibilité (TPR)** = {sensitivity:.2f}  
        La capacité du modèle à détecter les hausses correctes (vrais positifs).
        - **Spécificité (TNR)** = {specificity:.2f}  
        La capacité du modèle à éviter les alertes fausses (vrais négatifs).

        **Conseils d’interprétation** :
        - Plus l’AUC est proche de 1, meilleur est le pouvoir discriminant du modèle.
        - Le **seuil optimal** peut être ajusté selon votre tolérance au risque : 
        - Abaisser le seuil → détecte plus de hausses, mais plus de faux positifs.
        - Augmenter le seuil → réduit les fausses alertes, mais certaines hausses peuvent être manquées.
        - Analysez **sensibilité vs spécificité** pour équilibrer performance et sécurité des décisions.
""")


    else:
        st.warning("Vous devez d'abord entraîner les modèles !")
elif page == "💰 Prédictions":
    st.markdown("## 💰 Prédiction intelligente — Financial Advice")

    model_name = st.selectbox("Choisir le modèle pour prédiction :", list(st.session_state["best_models"].keys()))
    model = st.session_state["best_models"][model_name]

    st.markdown("### 📝 Saisir vos valeurs pour chaque feature")
    features_input = {}
    for f in ['Return', 'SMA_7', 'EMA_7', 'Volatility_7']:
        if f == "Volatility_7":
            features_input[f] = st.slider(f, 0.0, 0.1, float(df_ticker[f].iloc[-1]))
        else:
            features_input[f] = st.number_input(f, float(df_ticker[f].min()), float(df_ticker[f].max()), float(df_ticker[f].iloc[-1]))
    

    if st.button("🚀 Prédire et analyser"):
       
        input_features = np.array([list(features_input.values())]).reshape(1, -1)
        input_scaled = st.session_state["scaler"].transform(input_features)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        
        if prediction == 1:
            st.success(f"📈 Prédiction : HAUSSIER ({prediction_proba:.2f} de confiance)")
        else:
            st.warning(f"📉 Prédiction : BAISSIER ({1-prediction_proba:.2f} de confiance)")

        
        st.markdown("### 💡 Conseils personnalisés selon vos valeurs")
        for f, val in features_input.items():
            if f == "Return":
                if val > 0.05:
                    st.success(f"{f}: Rendement élevé — opportunité d'achat, mais surveillez la volatilité.")
                elif val < -0.05:
                    st.warning(f"{f}: Rendement négatif — attention aux pertes potentielles. Envisagez une stratégie de stop-loss.")
                else:
                    st.info(f"{f}: Rendement stable — marché neutre, envisager de conserver la position actuelle.")
            
            elif f == "Volatility_7":
                if val > 0.04:
                    st.warning(f"{f}: Forte volatilité — risque élevé, attention aux fluctuations brusques.")
                elif val > 0.02:
                    st.info(f"{f}: Volatilité modérée — opportunité pour traders expérimentés.")
                else:
                    st.success(f"{f}: Faible volatilité — risque réduit, idéal pour investissement sécurisé.")
            
            elif f in ["SMA_7", "EMA_7"]:
                current_price = df_ticker['Adj Close'].iloc[-1]
                trend = "haussière" if val > current_price else "baissière"
                st.info(f"{f}: Moyenne {trend} par rapport au prix actuel ({current_price:.2f}).")

       
        st.markdown("### 📌 Recommandations globales")
        if prediction == 1 and features_input['Volatility_7'] < 0.03:
            st.success("Le modèle prédit une hausse avec faible volatilité : envisagez d'acheter ou de renforcer votre position.")
        elif prediction == 1 and features_input['Volatility_7'] >= 0.03:
            st.info("Le modèle prédit une hausse, mais la volatilité est élevée : prudence, utilisez un stop-loss.")
        elif prediction == 0 and features_input['Volatility_7'] < 0.03:
            st.warning("Prévision baissière mais faible volatilité : envisagez de conserver ou sécuriser vos positions.")
        else:
            st.error("Prévision baissière avec forte volatilité : risque élevé, envisagez de vendre ou réduire votre exposition.")

        
        st.markdown("### 📊 Visualisation des valeurs saisies")
        st.bar_chart(pd.DataFrame.from_dict(features_input, orient='index', columns=['Valeur']))






st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569;'>© 2025 SmartInvest AI – Financial Analytics & Machine Learning</p>",
    unsafe_allow_html=True
)
