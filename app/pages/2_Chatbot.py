import re
import pandas as pd
import streamlit as st

from utils import (
    load_data,
    enrich_reviews,
    make_reviews_window,
    compute_station_metrics,
)

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot — Review Q&A (evidence-based)")

stations, reviews = load_data()
reviews_enriched = enrich_reviews(reviews)

# ----------------------------
# Controls
# ----------------------------
st.sidebar.header("Chat Controls")
window_days = st.sidebar.selectbox("Time window", [30, 90, 365], index=1)
min_snippets = st.sidebar.slider("Evidence snippets to show", 2, 8, 4, 1)

reviews_window, reviews_prior, cutoff, max_date = make_reviews_window(reviews_enriched, window_days)

st.caption(f"Answering using reviews from last {window_days} days (based on latest review date: {max_date.date()})")

# ----------------------------
# Helpers (local to this page)
# ----------------------------
def station_name(station_id: str) -> str:
    row = stations[stations["station_id"] == station_id]
    if row.empty:
        return station_id
    return row.iloc[0]["name"]

def join_station_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(stations[["station_id", "name", "address", "borough"]], on="station_id", how="left")

def format_snippet(row) -> str:
    # short evidence line
    date = row["review_date"].date() if pd.notnull(row["review_date"]) else ""
    rating = int(row["rating"]) if pd.notnull(row["rating"]) else ""
    text = (row["review_text"] or "").strip()
    if len(text) > 160:
        text = text[:160].rstrip() + "…"
    return f"- **{row['name']}** ({row['borough']}) — ⭐{rating} — {date}\n  “{text}”"

def top_stations_by_theme(theme: str, df: pd.DataFrame, min_mentions: int = 1, top_n: int = 5):
    themed = df[df["themes"].apply(lambda t: theme in t)].copy()
    if themed.empty:
        return pd.DataFrame(), themed

    counts = themed.groupby("station_id").size().reset_index(name="mentions")
    counts = counts[counts["mentions"] >= min_mentions].sort_values("mentions", ascending=False).head(top_n)
    counts = join_station_meta(counts)
    return counts, themed

def top_reasons_for_one_star(df: pd.DataFrame):
    ones = df[df["rating"] == 1].copy()
    if ones.empty:
        return pd.DataFrame(), ones
    # Count themes within 1-star reviews
    theme_counts = {}
    for themes in ones["themes"].tolist():
        for t in themes:
            theme_counts[t] = theme_counts.get(t, 0) + 1
    tc = pd.DataFrame([{"theme": k, "count": v} for k, v in theme_counts.items()]).sort_values("count", ascending=False)
    return tc, ones

def most_improved_stations(window_df: pd.DataFrame, prior_df: pd.DataFrame, top_n: int = 5):
    cur = compute_station_metrics(stations, window_df)
    prev = compute_station_metrics(stations, prior_df)

    comp = cur[["station_id", "name", "avg_rating", "neg_pct", "review_count"]].merge(
        prev[["station_id", "avg_rating", "neg_pct", "review_count"]],
        on="station_id",
        how="left",
        suffixes=("_cur", "_prev"),
    )

    comp["avg_rating_prev"] = comp["avg_rating_prev"].fillna(0.0)
    comp["neg_pct_prev"] = comp["neg_pct_prev"].fillna(0.0)

    comp["delta_rating"] = comp["avg_rating_cur"] - comp["avg_rating_prev"]
    comp["delta_neg_pct"] = comp["neg_pct_cur"] - comp["neg_pct_prev"]

    comp = comp[comp["review_count_cur"] > 0].copy()
    comp = comp.sort_values(["delta_rating", "review_count_cur"], ascending=[False, False]).head(top_n)
    return comp

def pick_snippets(df: pd.DataFrame, n: int):
    # prefer negative and 1-star first, then recent
    d = df.copy()
    # if sentiment_label exists, rank negatives first
    if "sentiment_label" in d.columns:
        rank = d["sentiment_label"].map({"negative": 0, "neutral": 1, "positive": 2}).fillna(1)
        d = d.assign(_rank=rank)
        d = d.sort_values(["_rank", "rating", "review_date"], ascending=[True, True, False])
    else:
        d = d.sort_values(["rating", "review_date"], ascending=[True, False])
    return join_station_meta(d.head(n))

def insufficient(msg: str):
    st.warning(msg)

# ----------------------------
# Query understanding (simple intent routing)
# ----------------------------
THEME_ALIASES = {
    "cleanliness": ["clean", "dirty", "filthy", "messy", "hygiene", "smell"],
    "staff": ["staff", "rude", "helpful", "cashier", "service"],
    "queues": ["queue", "line", "waiting", "wait", "slow", "crowded"],
    "pricing": ["price", "expensive", "overpriced", "cost"],
    "safety": ["safe", "unsafe", "security", "threat", "harass", "crime"],
    "toilets": ["toilet", "restroom", "bathroom", "soap", "loo"],
    "ev_charging": ["ev", "charger", "charging"],
    "car_wash": ["car wash", "jet wash", "vacuum"],
}

def detect_theme(q: str):
    ql = q.lower()
    for theme, words in THEME_ALIASES.items():
        if any(w in ql for w in words):
            return theme
    return None

def answer_question(question: str):
    q = question.strip()
    ql = q.lower()

    # 1) Cleanliness complaints / theme complaints
    if "complaint" in ql or "complaints" in ql or "mentions" in ql:
        theme = detect_theme(ql)
        if theme:
            counts, themed = top_stations_by_theme(theme, reviews_window, min_mentions=1, top_n=5)
            if counts.empty:
                insufficient(f"I couldn’t find enough mentions of **{theme}** in the last {window_days} days.")
                return

            st.markdown(f"### Top stations mentioning **{theme}** (last {window_days} days)")
            st.dataframe(counts[["name", "borough", "mentions"]], use_container_width=True)

            st.markdown("### Evidence")
            # show snippets from the theme, prioritizing low ratings
            evid = pick_snippets(themed, min_snippets)
            for _, row in evid.iterrows():
                st.write(format_snippet(row))
            return

    # 2) Top reasons for 1-star reviews
    if "1-star" in ql or "one star" in ql or "1 star" in ql:
        tc, ones = top_reasons_for_one_star(reviews_window)
        if ones.empty:
            insufficient(f"No 1-star reviews found in the last {window_days} days.")
            return

        st.markdown(f"### Top reasons/themes in **1-star** reviews (last {window_days} days)")
        if tc.empty:
            st.write("No themes detected in 1-star reviews (taxonomy didn’t match).")
        else:
            st.dataframe(tc.head(10), use_container_width=True)

        st.markdown("### Evidence (sample 1-star snippets)")
        evid = join_station_meta(ones.sort_values("review_date", ascending=False).head(min_snippets))
        for _, row in evid.iterrows():
            st.write(format_snippet(row))
        return

    # 3) Stations improved most
    if "improv" in ql or "improved" in ql or "improving" in ql:
        comp = most_improved_stations(reviews_window, reviews_prior, top_n=5)
        if comp.empty:
            insufficient("Not enough data to compute improvement vs the prior period.")
            return

        st.markdown(f"### Most improved stations (last {window_days} vs prior {window_days} days)")
        show = comp[["name", "delta_rating", "delta_neg_pct", "review_count_cur"]].copy()
        show["delta_neg_pct"] = show["delta_neg_pct"].apply(lambda x: f"{x*100:+.0f}%")
        st.dataframe(show, use_container_width=True)

        # evidence: show a few recent positive reviews from top station(s)
        top_station_ids = comp["station_id"].tolist()[:2]
        evid_src = reviews_window[reviews_window["station_id"].isin(top_station_ids)].copy()
        evid_src = evid_src.sort_values(["rating", "review_date"], ascending=[False, False])
        evid = join_station_meta(evid_src.head(min_snippets))
        st.markdown("### Evidence (recent higher-rated snippets from top improved stations)")
        for _, row in evid.iterrows():
            st.write(format_snippet(row))
        return

    # 4) Safety concerns
    if "safety" in ql or "unsafe" in ql or "security" in ql:
        theme = "safety"
        counts, themed = top_stations_by_theme(theme, reviews_window, min_mentions=1, top_n=5)
        if counts.empty:
            insufficient(f"No safety-related mentions found in the last {window_days} days.")
            return

        st.markdown(f"### Stations with recurring **safety** mentions (last {window_days} days)")
        st.dataframe(counts[["name", "borough", "mentions"]], use_container_width=True)

        st.markdown("### Evidence")
        evid = pick_snippets(themed, min_snippets)
        for _, row in evid.iterrows():
            st.write(format_snippet(row))
        return

    # 5) EV charging feedback
    if "ev" in ql or "charging" in ql or "charger" in ql:
        theme = "ev_charging"
        counts, themed = top_stations_by_theme(theme, reviews_window, min_mentions=1, top_n=5)
        if counts.empty:
            insufficient(f"No EV-charging mentions found in the last {window_days} days.")
            return

        st.markdown(f"### EV charging feedback (last {window_days} days)")
        st.dataframe(counts[["name", "borough", "mentions"]], use_container_width=True)

        st.markdown("### Evidence")
        evid = pick_snippets(themed, min_snippets)
        for _, row in evid.iterrows():
            st.write(format_snippet(row))
        return

    # fallback
    insufficient(
        "I can’t confidently answer that yet.\n\n"
        "Try questions like:\n"
        "- Which stations have the most complaints about cleanliness?\n"
        "- What are the top reasons for 1-star reviews?\n"
        "- Which stations improved the most in the last 90 days?\n"
        "- Are there recurring mentions of safety concerns?\n"
        "- Summarize common feedback about EV charging availability."
    )

# ----------------------------
# Chat UI (Streamlit chat)
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show prior messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about stations, themes, trends, complaints, 1-star reasons...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_question(prompt)

    # store a lightweight assistant "ack" in history (UI already displayed the full answer)
    st.session_state.chat_history.append({"role": "assistant", "content": "_Answered using review evidence above._"})
