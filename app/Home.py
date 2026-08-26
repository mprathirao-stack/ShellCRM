import pandas as pd
import streamlit as st

from utils import (
    load_data, enrich_reviews, compute_station_metrics, compute_overall_summary, top_themes_from,
    theme_deltas, detect_spikes
)

st.set_page_config(page_title="Shell London Reviews", layout="wide")
st.title("Shell London Reviews — Executive Summary")

stations, reviews = load_data()
reviews_enriched = enrich_reviews(reviews)

st.sidebar.header("Summary Controls")
time_window_days = st.sidebar.selectbox("Time window", [30, 90, 365], index=1)

max_date = reviews_enriched["review_date"].max()
cutoff = max_date - pd.Timedelta(days=int(time_window_days))
reviews_window = reviews_enriched[reviews_enriched["review_date"] >= cutoff].copy()

summary = compute_overall_summary(reviews_window)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Reviews (window)", summary["reviews"])
with c2:
    st.metric("Avg rating", f"{summary['avg_rating']:.2f}" if summary["reviews"] > 0 else "N/A")
with c3:
    st.metric("Negative %", f"{round(summary['neg_pct']*100):.0f}%")
with c4:
    st.metric("Stations covered", stations.shape[0])

st.write("### Sentiment split (window)")
st.write(f"✅ Positive: **{summary['pos']}**   |   😐 Neutral: **{summary['neu']}**   |   ❌ Negative: **{summary['neg']}**")

st.write("### Main experience drivers (themes)")
pos_reviews = reviews_window[reviews_window["sentiment_label"] == "positive"]
neg_reviews = reviews_window[reviews_window["sentiment_label"] == "negative"]

top_pos = top_themes_from(pos_reviews, n=6)
top_neg = top_themes_from(neg_reviews, n=6)

colA, colB = st.columns(2)
with colA:
    st.markdown("**Top themes in positive reviews**")
    if not top_pos:
        st.caption("No positive themes detected.")
    else:
        for theme, cnt in top_pos:
            st.write(f"- {theme} ({cnt})")

with colB:
    st.markdown("**Top themes in negative reviews**")
    if not top_neg:
        st.caption("No negative themes detected.")
    else:
        for theme, cnt in top_neg:
            st.write(f"- {theme} ({cnt})")

st.write("### Trend vs previous period")
prior_start = cutoff - pd.Timedelta(days=int(time_window_days))
prior_end = cutoff

reviews_prior = reviews_enriched[
    (reviews_enriched["review_date"] >= prior_start) &
    (reviews_enriched["review_date"] < prior_end)
].copy()

cur = compute_overall_summary(reviews_window)
prev = compute_overall_summary(reviews_prior)

delta_rating = cur["avg_rating"] - prev["avg_rating"] if prev["reviews"] > 0 else 0.0
delta_neg = (cur["neg_pct"] - prev["neg_pct"]) if prev["reviews"] > 0 else 0.0

t1, t2, t3 = st.columns(3)
with t1:
    st.metric("Avg rating change", f"{delta_rating:+.2f}")
with t2:
    st.metric("Negative % change", f"{round(delta_neg*100):+.0f}%")
with t3:
    st.metric("Prior period reviews", prev["reviews"])

st.write("### Rising / falling themes vs prior period")
neg_window = reviews_window[reviews_window["sentiment_label"] == "negative"]
neg_prior = reviews_prior[reviews_prior["sentiment_label"] == "negative"]
pos_window = reviews_window[reviews_window["sentiment_label"] == "positive"]
pos_prior = reviews_prior[reviews_prior["sentiment_label"] == "positive"]

complaint_deltas = theme_deltas(neg_window, neg_prior)
praise_deltas = theme_deltas(pos_window, pos_prior)

rC, rD = st.columns(2)
with rC:
    st.markdown("**Rising complaints** (negative mentions, current vs prior period)")
    rising_complaints = complaint_deltas[complaint_deltas["delta"] > 0]
    if rising_complaints.empty:
        st.caption("No themes showing a rise in complaints.")
    else:
        st.dataframe(rising_complaints, use_container_width=True, hide_index=True)

with rD:
    st.markdown("**Rising praise** (positive mentions, current vs prior period)")
    rising_praise = praise_deltas[praise_deltas["delta"] > 0]
    if rising_praise.empty:
        st.caption("No themes showing a rise in praise.")
    else:
        st.dataframe(rising_praise, use_container_width=True, hide_index=True)

st.write("### Notable spikes")
spikes = detect_spikes(stations, reviews_enriched, max_date, spike_days=30, baseline_days=180)
if spikes.empty:
    st.caption("No stations show a sudden spike in negative reviews right now.")
else:
    show_spikes = spikes[["name", "borough", "top_theme", "spike_neg_count", "baseline_neg_count", "ratio"]].copy()
    show_spikes["ratio"] = show_spikes["ratio"].apply(lambda x: "∞" if x == float("inf") else f"{x:.1f}x")
    show_spikes = show_spikes.rename(columns={
        "top_theme": "driven by",
        "spike_neg_count": "negative reviews (last 30d)",
        "baseline_neg_count": "negative reviews (prior 180d, per 30d)",
        "ratio": "vs baseline",
    })
    st.dataframe(show_spikes, use_container_width=True, hide_index=True)
    st.caption("Flagged when negative reviews in the last 30 days run well above the station's normal rate.")

st.write("### Stations improving vs deteriorating")
stations_cur = compute_station_metrics(stations, reviews_window)
stations_prev = compute_station_metrics(stations, reviews_prior)

compare = stations_cur[["station_id", "name", "avg_rating", "neg_pct", "review_count"]].merge(
    stations_prev[["station_id", "avg_rating", "neg_pct", "review_count"]],
    on="station_id",
    how="left",
    suffixes=("_cur", "_prev"),
)

compare["avg_rating_prev"] = compare["avg_rating_prev"].fillna(0.0)
compare["neg_pct_prev"] = compare["neg_pct_prev"].fillna(0.0)

compare["delta_rating"] = compare["avg_rating_cur"] - compare["avg_rating_prev"]
compare["delta_neg_pct"] = compare["neg_pct_cur"] - compare["neg_pct_prev"]

compare = compare[compare["review_count_cur"] > 0].copy()

best = compare.sort_values("delta_rating", ascending=False).head(5)
worst = compare.sort_values("delta_rating", ascending=True).head(5)

cA, cB = st.columns(2)
with cA:
    st.markdown("**Most improved (avg rating)**")
    st.dataframe(best[["name", "delta_rating", "review_count_cur"]], use_container_width=True)

with cB:
    st.markdown("**Most deteriorated (avg rating)**")
    st.dataframe(worst[["name", "delta_rating", "review_count_cur"]], use_container_width=True)
