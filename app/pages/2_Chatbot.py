import re
import pandas as pd
import streamlit as st

from utils import (
    load_data,
    enrich_reviews,
    make_reviews_window,
    compute_station_metrics,
    compute_overall_summary,
    top_themes_from,
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
def join_station_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(stations[["station_id", "name", "address", "borough", "region"]], on="station_id", how="left")

def format_snippet(row) -> str:
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

def pick_snippets(df: pd.DataFrame, n: int, prefer: str = "negative"):
    d = df.copy()
    if "sentiment_label" in d.columns:
        order = {"negative": 0, "neutral": 1, "positive": 2}
        if prefer == "positive":
            order = {"positive": 0, "neutral": 1, "negative": 2}
        rank = d["sentiment_label"].map(order).fillna(1)
        d = d.assign(_rank=rank)
        rating_ascending = prefer != "positive"
        d = d.sort_values(["_rank", "rating", "review_date"], ascending=[True, rating_ascending, False])
    else:
        d = d.sort_values(["rating", "review_date"], ascending=[True, False])
    return join_station_meta(d.head(n))

def insufficient(msg: str):
    st.warning(msg)

def render_table(df: pd.DataFrame, cols: list):
    st.dataframe(df[cols], use_container_width=True)

# ----------------------------
# Station name recognition
# ----------------------------
def find_station_mentions(ql: str):
    matches = []
    for _, srow in stations.iterrows():
        name = srow["name"].lower()
        short = name.replace("shell ", "").strip()
        if name in ql or (short and len(short) > 3 and short in ql):
            matches.append(srow["station_id"])
    return matches

# ----------------------------
# Theme taxonomy
# ----------------------------
THEME_ALIASES = {
    "cleanliness": ["clean", "dirty", "filthy", "messy", "hygiene", "smell"],
    "staff": ["staff", "rude", "helpful", "cashier", "attendant"],
    "queues": ["queue", "line", "waiting", "wait", "slow", "crowded"],
    "pricing": ["price", "pricing", "expensive", "overpriced", "cost"],
    "safety": ["safe", "unsafe", "security", "threat", "harass", "crime"],
    "toilets": ["toilet", "restroom", "bathroom", "soap", "loo"],
    "ev_charging": ["ev ", "ev charging", "charger", "charging point", "charging"],
    "car_wash": ["car wash", "jet wash", "vacuum"],
}

def detect_theme(ql: str):
    for theme, words in THEME_ALIASES.items():
        if any(w in ql for w in words):
            return theme
    return None

# ----------------------------
# Ranking questions ("best station", "most reviewed", etc.)
# ----------------------------
RANKING_PHRASES = {
    "best_rated": ["best rated", "highest rated", "best station", "top rated", "best rewarded",
                   "highest rating", "top station", "highly rated", "favourite station", "favorite station"],
    "worst_rated": ["worst rated", "lowest rated", "worst station", "poorest rated", "lowest rating"],
    "most_reviewed": ["most reviewed", "most popular", "busiest", "most visited", "highest volume", "most reviews"],
    "least_reviewed": ["least reviewed", "fewest reviews", "lowest volume", "least popular"],
}

def detect_ranking(ql: str):
    for key, phrases in RANKING_PHRASES.items():
        if any(p in ql for p in phrases):
            return key
    if ("best" in ql or "top" in ql or "highest" in ql) and ("station" in ql or "rated" in ql or "reward" in ql or "rating" in ql):
        return "best_rated"
    if ("worst" in ql or "lowest" in ql) and ("station" in ql or "rated" in ql or "rating" in ql):
        return "worst_rated"
    return None

def render_ranking(kind: str):
    metrics = compute_station_metrics(stations, reviews_window)
    qualified = metrics[metrics["review_count"] > 0].copy()
    if qualified.empty:
        insufficient(f"No reviews in the last {window_days} days to rank stations by.")
        return

    if kind == "best_rated":
        title, sort_col, ascending = "Highest-rated stations", "avg_rating", False
    elif kind == "worst_rated":
        title, sort_col, ascending = "Lowest-rated stations", "avg_rating", True
    elif kind == "most_reviewed":
        title, sort_col, ascending = "Most-reviewed stations", "review_count", False
    else:
        title, sort_col, ascending = "Least-reviewed stations", "review_count", True

    ranked = qualified.sort_values([sort_col, "review_count"], ascending=[ascending, ascending]).head(5)
    st.markdown(f"### {title} (last {window_days} days)")
    render_table(ranked, ["name", "borough", "region", "avg_rating", "review_count", "neg_pct_display"])

    top_id = ranked.iloc[0]["station_id"]
    prefer = "positive" if kind in ("best_rated", "most_reviewed") else "negative"
    evid_src = reviews_window[reviews_window["station_id"] == top_id]
    st.markdown(f"### Evidence from {ranked.iloc[0]['name']}")
    for _, row in pick_snippets(evid_src, min_snippets, prefer=prefer).iterrows():
        st.write(format_snippet(row))

# ----------------------------
# Improving / deteriorating stations
# ----------------------------
def station_trend_comparison(top_n: int = 5):
    cur = compute_station_metrics(stations, reviews_window)
    prev = compute_station_metrics(stations, reviews_prior)

    comp = cur[["station_id", "name", "borough", "region", "avg_rating", "neg_pct", "review_count"]].merge(
        prev[["station_id", "avg_rating", "neg_pct", "review_count"]],
        on="station_id",
        how="left",
        suffixes=("_cur", "_prev"),
    )

    comp["avg_rating_prev"] = comp["avg_rating_prev"].fillna(0.0)
    comp["neg_pct_prev"] = comp["neg_pct_prev"].fillna(0.0)
    comp["delta_rating"] = comp["avg_rating_cur"] - comp["avg_rating_prev"]
    comp["delta_neg_pct"] = comp["neg_pct_cur"] - comp["neg_pct_prev"]
    return comp[comp["review_count_cur"] > 0].copy()

def render_trend(direction: str):
    comp = station_trend_comparison()
    if comp.empty:
        insufficient("Not enough data to compare against the prior period.")
        return

    ascending = direction == "deteriorated"
    ranked = comp.sort_values(["delta_rating", "review_count_cur"], ascending=[ascending, False]).head(5)

    label = "Most improved" if direction == "improved" else "Most deteriorated"
    st.markdown(f"### {label} stations (last {window_days} vs prior {window_days} days)")
    show = ranked[["name", "borough", "delta_rating", "delta_neg_pct", "review_count_cur"]].copy()
    show["delta_neg_pct"] = show["delta_neg_pct"].apply(lambda x: f"{x*100:+.0f}%")
    render_table(show, list(show.columns))

    top_ids = ranked["station_id"].tolist()[:2]
    evid_src = reviews_window[reviews_window["station_id"].isin(top_ids)]
    prefer = "positive" if direction == "improved" else "negative"
    st.markdown("### Evidence")
    for _, row in pick_snippets(evid_src, min_snippets, prefer=prefer).iterrows():
        st.write(format_snippet(row))

# ----------------------------
# Overall summary
# ----------------------------
def render_overall_summary(scope_df: pd.DataFrame, scope_label: str):
    summary = compute_overall_summary(scope_df)
    if summary["reviews"] == 0:
        insufficient(f"No reviews found for {scope_label} in the last {window_days} days.")
        return

    st.markdown(f"### Overall picture — {scope_label} (last {window_days} days)")
    st.write(
        f"**{summary['reviews']}** reviews, average rating **{summary['avg_rating']:.2f}** — "
        f"✅ {summary['pos']} positive · 😐 {summary['neu']} neutral · ❌ {summary['neg']} negative "
        f"({round(summary['neg_pct']*100)}% negative)."
    )

    pos_themes = top_themes_from(scope_df[scope_df["sentiment_label"] == "positive"], n=5)
    neg_themes = top_themes_from(scope_df[scope_df["sentiment_label"] == "negative"], n=5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top praised themes**")
        st.write(", ".join(f"{t} ({c})" for t, c in pos_themes) if pos_themes else "None detected")
    with c2:
        st.markdown("**Top complaint themes**")
        st.write(", ".join(f"{t} ({c})" for t, c in neg_themes) if neg_themes else "None detected")

# ----------------------------
# Region / borough detection
# ----------------------------
def detect_region_or_borough(ql: str):
    for region in stations["region"].dropna().unique():
        if region.lower() in ql:
            return "region", region
    for borough in stations["borough"].dropna().unique():
        if borough.lower() in ql:
            return "borough", borough
    return None, None

# ----------------------------
# Main intent router
# ----------------------------
def answer_question(question: str):
    q = question.strip()
    ql = q.lower()

    # 1) Named station -> full station overview, optionally theme-scoped
    mentioned = find_station_mentions(ql)
    if mentioned:
        sid = mentioned[0]
        srow = stations[stations["station_id"] == sid].iloc[0]
        srevs = reviews_window[reviews_window["station_id"] == sid]
        if srevs.empty:
            insufficient(f"No reviews for **{srow['name']}** in the last {window_days} days. Try a longer time window.")
            return

        metrics_row = compute_station_metrics(stations, reviews_window)
        metrics_row = metrics_row[metrics_row["station_id"] == sid].iloc[0]

        st.markdown(f"### {srow['name']} — {srow['borough']} ({srow['region']})")
        st.write(
            f"Avg rating **{metrics_row['avg_rating_display']}** from **{metrics_row['review_count']}** reviews "
            f"(last {window_days} days) — ✅ {metrics_row['pos_count']} · 😐 {metrics_row['neu_count']} · "
            f"❌ {metrics_row['neg_count']} ({metrics_row['neg_pct_display']} negative)."
        )

        theme = detect_theme(ql)
        if theme:
            themed = srevs[srevs["themes"].apply(lambda t: theme in t)]
            if themed.empty:
                st.write(f"No **{theme}** mentions found for this station in this window.")
            else:
                st.markdown(f"**Evidence on {theme}:**")
                for _, row in pick_snippets(themed, min_snippets).iterrows():
                    st.write(format_snippet(row))
        else:
            top_themes = top_themes_from(srevs, n=5)
            st.write("**Key themes:** " + (", ".join(f"{t} ({c})" for t, c in top_themes) if top_themes else "None detected"))
            pos = srevs[srevs["rating"] >= 4].sort_values("review_date", ascending=False).head(3)
            neg = srevs[srevs["rating"] <= 2].sort_values("review_date", ascending=False).head(3)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top positives**")
                if pos.empty:
                    st.caption("No 4-5 star reviews in this window.")
                for _, row in pos.iterrows():
                    st.write(f"⭐{int(row['rating'])} — {row['review_date'].date()} — “{row['review_text']}”")
            with c2:
                st.markdown("**Top negatives**")
                if neg.empty:
                    st.caption("No 1-2 star reviews in this window.")
                for _, row in neg.iterrows():
                    st.write(f"⭐{int(row['rating'])} — {row['review_date'].date()} — “{row['review_text']}”")
        return

    # 2) Ranking questions (best/worst/most-reviewed/least-reviewed)
    ranking = detect_ranking(ql)
    if ranking:
        render_ranking(ranking)
        return

    # 3) Improving / deteriorating trend questions
    if any(w in ql for w in ["improv", "getting better", "turnaround", "recovered"]):
        render_trend("improved")
        return
    if any(w in ql for w in ["deteriorat", "worsen", "declin", "getting worse", "dropping", "sliding", "regressed"]):
        render_trend("deteriorated")
        return

    # 4) N-star review reasons (generalized from "1-star" to any star rating)
    star_match = re.search(r"\b([1-5])\s*-?\s*star", ql)
    if star_match:
        star = int(star_match.group(1))
        subset = reviews_window[reviews_window["rating"] == star]
        if subset.empty:
            insufficient(f"No {star}-star reviews found in the last {window_days} days.")
            return

        theme_counts = {}
        for themes in subset["themes"].tolist():
            for t in themes:
                theme_counts[t] = theme_counts.get(t, 0) + 1
        tc = pd.DataFrame([{"theme": k, "count": v} for k, v in theme_counts.items()]).sort_values(
            "count", ascending=False
        )

        st.markdown(f"### Top reasons/themes in **{star}-star** reviews (last {window_days} days)")
        if tc.empty:
            st.write("No recurring themes detected in these reviews (taxonomy didn't match).")
        else:
            render_table(tc.head(10), list(tc.columns))

        st.markdown(f"### Evidence (sample {star}-star snippets)")
        evid = join_station_meta(subset.sort_values("review_date", ascending=False).head(min_snippets))
        for _, row in evid.iterrows():
            st.write(format_snippet(row))
        return

    # 5) Region / borough summary
    kind, place = detect_region_or_borough(ql)
    if place:
        subset_ids = stations[stations[kind] == place]["station_id"]
        subset_reviews = reviews_window[reviews_window["station_id"].isin(subset_ids)]
        render_overall_summary(subset_reviews, place)
        metrics = compute_station_metrics(stations[stations[kind] == place], subset_reviews)
        st.markdown(f"**Stations in {place}:**")
        render_table(
            metrics.sort_values("avg_rating", ascending=False),
            ["name", "borough", "avg_rating", "review_count"],
        )
        return

    # 6) Theme-based question (complaints, praise, or general feedback about a theme)
    theme = detect_theme(ql)
    if theme:
        neg_cue = any(w in ql for w in ["complain", "complaint", "issue", "problem", "bad", "negative", "concern", "worst"])
        pos_cue = any(w in ql for w in ["praise", "positive feedback", "good", "great", "recommend", "love"])

        if neg_cue and not pos_cue:
            focus_df, focus_word, prefer = reviews_window[reviews_window["sentiment_label"] == "negative"], "negative", "negative"
        elif pos_cue and not neg_cue:
            focus_df, focus_word, prefer = reviews_window[reviews_window["sentiment_label"] == "positive"], "positive", "positive"
        else:
            focus_df, focus_word, prefer = reviews_window, "overall", "negative"

        counts, themed = top_stations_by_theme(theme, focus_df, min_mentions=1, top_n=5)
        if counts.empty:
            insufficient(f"I couldn't find {focus_word} mentions of **{theme}** in the last {window_days} days.")
            return

        st.markdown(f"### Stations with {focus_word} mentions of **{theme}** (last {window_days} days)")
        render_table(counts, ["name", "borough", "mentions"])

        st.markdown("### Evidence")
        for _, row in pick_snippets(themed, min_snippets, prefer=prefer).iterrows():
            st.write(format_snippet(row))
        return

    # 7) Overall / general sentiment questions
    if any(w in ql for w in ["overall", "in general", "how are we doing", "general sentiment", "summary",
                              "how is shell doing", "how's it going", "how are things"]):
        render_overall_summary(reviews_window, "all London stations")
        return

    # 8) Best-effort fallback: still surface real data instead of a flat refusal
    st.info(
        "I couldn't match that to a specific question pattern, so here's the current overall picture instead — "
        "try naming a station, theme, or borough for a more targeted answer."
    )
    render_overall_summary(reviews_window, "all London stations")
    st.markdown("**Try asking things like:**")
    st.write(
        "- Which stations have the most complaints about cleanliness?\n"
        "- What are the top reasons for 1-star reviews?\n"
        "- Which stations improved the most in the last 90 days?\n"
        "- Which station has the best rating?\n"
        "- How are stations in East London doing?\n"
        "- Are there recurring mentions of safety concerns?\n"
        "- Summarize common feedback about EV charging availability."
    )

# ----------------------------
# Chat UI (Streamlit chat)
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about stations, themes, trends, complaints, ratings, boroughs...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_question(prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": "_Answered using review evidence above._"})
