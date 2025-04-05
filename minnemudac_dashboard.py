# minnemudac_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MinneMUDAC 2025 Dashboard", layout="wide")

st.title("🧠 BBBS Match Risk, Sentiment & Prediction Explorer")

# --------------- LOAD DATA ---------------
@st.cache_data
def load_data():
    predictions = pd.read_csv("test_risk_predictions.csv")
    notes = pd.read_csv("training_data.csv")
    submission = pd.read_csv("Testset_Predictions_Submit.csv")
    return predictions, notes, submission

pred_df, notes_df, submit_template = load_data()
notes_df["Completion Date"] = pd.to_datetime(notes_df["Completion Date"], errors="coerce")

# Tabs layout
tabs = st.tabs(["📋 Summary", "📈 Match Insights", "🔤 Text Analysis", "🧬 Modeling", "📁 Submission"])

# --------------- TAB 1: Summary ---------------
with tabs[0]:
    st.header("📋 Match Summary & Filters")
    min_days = st.slider("Minimum Days Active", 0, int(pred_df["Note Duration (days)"].max()), 0)
    show_risk = st.checkbox("Show Only At-Risk Matches", value=False)

    filtered = pred_df[pred_df["Note Duration (days)"] >= min_days]
    if show_risk:
        filtered = filtered[filtered["Predicted At Risk"] == 1]

    st.dataframe(filtered, use_container_width=True)

    st.subheader("🔮 Predicted Match Lengths")
    fig, ax = plt.subplots()
    sns.histplot(filtered['Predicted Match Length'], bins=20, kde=True, ax=ax)
    ax.set_title("Forecasted Match Durations")
    st.pyplot(fig)

# --------------- TAB 2: Match Insights ---------------
with tabs[1]:
    st.header("📈 Match Sentiment Timeline")
    match_ids = filtered["Match ID 18Char"].unique()
    match_id = st.selectbox("Choose a Match ID", match_ids)

    match_notes = notes_df[notes_df["Match ID 18Char"] == match_id].copy()
    match_notes.sort_values("Completion Date", inplace=True)
    match_notes["Sentiment"] = match_notes["Match Support Contact Notes"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=match_notes, x="Completion Date", y="Sentiment", marker="o", ax=ax)
    plt.axhline(0, linestyle="--", color="gray")
    ax.set_title(f"Sentiment Timeline for Match {match_id}")
    st.pyplot(fig)

    st.subheader("📉 Early vs. Late Sentiment Shift")
    if len(match_notes) >= 2:
        half = len(match_notes) // 2
        shift = match_notes["Sentiment"].iloc[half:].mean() - match_notes["Sentiment"].iloc[:half].mean()
        st.metric("Change in Sentiment", f"{shift:.3f}", delta_color="inverse")

# --------------- TAB 3: Text Analysis ---------------
with tabs[2]:
    st.header("🔤 Textual Features & Keywords")
    all_notes = notes_df["Match Support Contact Notes"].dropna().astype(str)
    full_text = " ".join(all_notes)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_text)

    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud from Match Notes")
    st.pyplot(plt)

    st.markdown("Most common words show recurring themes in closure and support notes.")

# --------------- TAB 4: Modeling ---------------
with tabs[3]:
    st.header("🧬 Predictive Modeling & SHAP Values")
    # Assume model already trained (use dummy data here for display)
    X = pred_df[["Call Span", "Big Age Initially", "Little Age Initially"]].dropna()
    y = pred_df.loc[X.index, "Predicted Match Length"]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("📊 SHAP Summary Plot")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X)
    st.pyplot()

# --------------- TAB 5: Submission ---------------
with tabs[4]:
    st.header("📁 Submit to MinneMUDAC")
    result_df = pd.read_csv("Predictions.csv")
    submit_template.columns = submit_template.columns.str.strip()
    result_df.rename(columns={"Match ID": "MatchID18Char"}, inplace=True)
    merged = submit_template.merge(result_df, on="MatchID18Char", how="left")
    merged["YourTeamID"] = "U33"
    merged["PredictedMatchLength"] = merged["Predicted Match Length"]
    merged.drop(columns=["Predicted Match Length"], inplace=True)

    st.download_button("📥 Download Final Submission", data=merged.to_csv(index=False),
                       file_name="Testset_Predictions_Submit_Final.csv", mime="text/csv")

    st.success("Ready to upload to the competition portal!")
