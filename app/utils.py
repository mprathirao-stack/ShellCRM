import pandas as pd
import streamlit as st
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

# ----------------------------
# Theme taxonomy (editable)
# ----------------------------
THEME_KEYWORDS = {
    "cleanliness": ["clean", "dirty", "filthy", "messy", "sticky", "smell", "smelly", "hygiene", "grime"],
    "staff": [
        "staff", "cashier", "attendant", "rude", "polite", "helpful", "unhelpful",
        "friendly", "customer service", "service"
    ],
    "queues": ["queue", "queues", "line", "waiting", "wait", "slow", "crowded", "rush"],
    "pricing": ["price", "prices", "expensive", "cost", "overpriced", "rip off", "ripoff"],
    "safety": [
        "unsafe", "safe", "security", "harass", "harassment", "threat", "threatening",
        "crime", "scary", "danger"
    ],
    "toilets": ["toilet", "toilets", "restroom", "bathroom", "loo", "washroom", "soap"],
    "ev_charging": ["ev", "charger", "charging", "charge point", "chargepoint", "rapid charger", "broken charger"],
    "car_wash": ["car wash", "jet wash", "wash", "vacuum"],
}

@st.cache_resource
def get_vader():
    # Ensure VADER lexicon exists in the deployment environment (Streamlit Cloud)
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    return SentimentIntensityAnalyzer()


def tag_themes(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    t = text.lower()
    found = []
    for theme, keywords in THEME_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                found.append(theme)
                break
    return found

def vader_sentiment_label(text: str) -> tuple[str, float]:
    if not isinstance(text, str) or not text.strip():
        return ("neutral", 0.0)

    sia = get_vader()
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.20:
        return ("positive", score)
    if score <= -0.20:
        return ("negative", score)
    return ("neutral", score)

@st.cache_data
def load_data():
    stations = pd.read_csv("data/stations.csv")
    reviews = pd.read_csv("data/reviews.csv", parse_dates=["review_date"])

    stations["station_id"] = stations["station_id"].astype(str).str.strip()
    reviews["station_id"] = reviews["station_id"].astype(str).str.strip()
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")

    return stations, reviews

def enrich_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    out = reviews_df.copy()
    out["themes"] = out["review_text"].apply(tag_themes)

    sent = out["review_text"].apply(vader_sentiment_label)
    out["sentiment_label"] = sent.apply(lambda x: x[0])
    out["sentiment_score"] = sent.apply(lambda x: x[1])
    return out

def compute_overall_summary(reviews_df: pd.DataFrame) -> dict:
    if reviews_df.empty:
        return {"reviews": 0, "avg_rating": 0.0, "neg_pct": 0.0, "pos": 0, "neu": 0, "neg": 0}

    total = len(reviews_df)
    avg_rating = float(reviews_df["rating"].mean())

    pos = int((reviews_df["sentiment_label"] == "positive").sum())
    neu = int((reviews_df["sentiment_label"] == "neutral").sum())
    neg = int((reviews_df["sentiment_label"] == "negative").sum())

    neg_pct = neg / total if total > 0 else 0.0

    return {"reviews": total, "avg_rating": avg_rating, "neg_pct": neg_pct, "pos": pos, "neu": neu, "neg": neg}

def compute_station_metrics(stations: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        out = stations.copy()
        out["review_count"] = 0
        out["avg_rating"] = 0.0
        out["pos_count"] = 0
        out["neu_count"] = 0
        out["neg_count"] = 0
        out["neg_pct"] = 0.0
        out["avg_rating_display"] = "N/A"
        out["review_count_display"] = "0"
        out["neg_pct_display"] = "0%"
        return out

    tmp = reviews.copy()
    tmp["pos"] = (tmp["sentiment_label"] == "positive").astype(int)
    tmp["neu"] = (tmp["sentiment_label"] == "neutral").astype(int)
    tmp["neg"] = (tmp["sentiment_label"] == "negative").astype(int)

    agg = (
        tmp.groupby("station_id")
        .agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean"),
            pos_count=("pos", "sum"),
            neu_count=("neu", "sum"),
            neg_count=("neg", "sum"),
        )
        .reset_index()
    )

    out = stations.merge(agg, on="station_id", how="left")
    out["review_count"] = out["review_count"].fillna(0).astype(int)
    out["avg_rating"] = out["avg_rating"].fillna(0.0)

    out["pos_count"] = out["pos_count"].fillna(0).astype(int)
    out["neu_count"] = out["neu_count"].fillna(0).astype(int)
    out["neg_count"] = out["neg_count"].fillna(0).astype(int)

    out["neg_pct"] = out.apply(
        lambda r: (r["neg_count"] / r["review_count"]) if r["review_count"] > 0 else 0.0,
        axis=1
    )

    out["avg_rating_display"] = out["avg_rating"].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    out["review_count_display"] = out["review_count"].astype(str)
    out["neg_pct_display"] = out["neg_pct"].apply(lambda x: f"{round(x*100):.0f}%")

    return out

def top_themes_from(df: pd.DataFrame, n: int = 6):
    all_t = []
    for themes in df["themes"].tolist():
        all_t.extend(themes)
    return Counter(all_t).most_common(n)

def theme_deltas(window_df: pd.DataFrame, prior_df: pd.DataFrame) -> pd.DataFrame:
    """Compare theme mention counts between two periods (e.g. negative reviews now vs before)."""
    def counts(df):
        c = Counter()
        for themes in df["themes"].tolist():
            c.update(themes)
        return c

    cur = counts(window_df)
    prev = counts(prior_df)
    all_themes = set(cur) | set(prev)

    rows = [
        {"theme": t, "current": cur.get(t, 0), "prior": prev.get(t, 0), "delta": cur.get(t, 0) - prev.get(t, 0)}
        for t in all_themes
    ]
    if not rows:
        return pd.DataFrame(columns=["theme", "current", "prior", "delta"])
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def detect_spikes(
    stations: pd.DataFrame,
    reviews_enriched: pd.DataFrame,
    max_date: pd.Timestamp,
    spike_days: int = 30,
    baseline_days: int = 180,
    min_count: int = 3,
    ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Flags stations with a sudden recent increase in negative reviews:
    negative reviews in the last `spike_days` compared against the
    per-`spike_days` negative review rate over the preceding `baseline_days`.
    """
    spike_start = max_date - pd.Timedelta(days=spike_days)
    baseline_start = spike_start - pd.Timedelta(days=baseline_days)

    neg = reviews_enriched[reviews_enriched["sentiment_label"] == "negative"]
    spike_df = neg[neg["review_date"] > spike_start]
    baseline_df = neg[(neg["review_date"] > baseline_start) & (neg["review_date"] <= spike_start)]

    spike_counts = spike_df.groupby("station_id").size().rename("spike_neg_count")
    baseline_counts = baseline_df.groupby("station_id").size().rename("baseline_neg_count")

    combined = pd.concat([spike_counts, baseline_counts], axis=1).fillna(0).reset_index()
    if combined.empty:
        return pd.DataFrame(columns=["station_id", "name", "borough", "spike_neg_count", "baseline_neg_count", "ratio", "top_theme"])

    windows_in_baseline = baseline_days / spike_days
    combined["baseline_rate"] = combined["baseline_neg_count"] / windows_in_baseline

    def ratio(row):
        if row["baseline_rate"] > 0:
            return row["spike_neg_count"] / row["baseline_rate"]
        return float("inf") if row["spike_neg_count"] >= min_count else 0.0

    combined["ratio"] = combined.apply(ratio, axis=1)

    flagged = combined[
        (combined["spike_neg_count"] >= min_count) & (combined["ratio"] >= ratio_threshold)
    ].copy()

    if flagged.empty:
        return pd.DataFrame(columns=["station_id", "name", "borough", "spike_neg_count", "baseline_neg_count", "ratio", "top_theme"])

    top_theme = {}
    for sid in flagged["station_id"]:
        themes = []
        for t in spike_df.loc[spike_df["station_id"] == sid, "themes"]:
            themes.extend(t)
        top_theme[sid] = Counter(themes).most_common(1)[0][0] if themes else None
    flagged["top_theme"] = flagged["station_id"].map(top_theme)

    flagged = flagged.merge(stations[["station_id", "name", "borough"]], on="station_id", how="left")
    flagged["spike_neg_count"] = flagged["spike_neg_count"].astype(int)
    flagged["baseline_neg_count"] = flagged["baseline_neg_count"].astype(int)

    return flagged.sort_values("ratio", ascending=False).reset_index(drop=True)


def make_reviews_window(reviews_enriched: pd.DataFrame, window_days: int):
    """
    Returns (reviews_window, reviews_prior, cutoff, max_date).
    Window is based on latest review date in data (stable for demo).
    """
    max_date = reviews_enriched["review_date"].max()
    cutoff = max_date - pd.Timedelta(days=int(window_days))

    reviews_window = reviews_enriched[reviews_enriched["review_date"] >= cutoff].copy()

    prior_start = cutoff - pd.Timedelta(days=int(window_days))
    prior_end = cutoff
    reviews_prior = reviews_enriched[
        (reviews_enriched["review_date"] >= prior_start) &
        (reviews_enriched["review_date"] < prior_end)
    ].copy()

    return reviews_window, reviews_prior, cutoff, max_date

