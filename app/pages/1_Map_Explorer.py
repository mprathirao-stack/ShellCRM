import pandas as pd
import streamlit as st
import pydeck as pdk

from collections import Counter
from utils import load_data, enrich_reviews, compute_station_metrics

st.set_page_config(page_title="Map Explorer", layout="wide")
st.title("Map Explorer")

stations, reviews = load_data()
reviews_enriched = enrich_reviews(reviews)

# Sidebar filters
st.sidebar.header("Filters")
time_window_days = st.sidebar.selectbox("Time window", [30, 90, 365], index=1)
min_reviews = st.sidebar.slider("Minimum review count", 0, 200, 0, 5)
rating_range = st.sidebar.slider("Avg rating range", 1.0, 5.0, (1.0, 5.0), 0.1)

boroughs = ["All"] + sorted(stations["borough"].dropna().unique().tolist())
borough_filter = st.sidebar.selectbox("Borough", boroughs, index=0)

# Filter reviews by window
max_date = reviews_enriched["review_date"].max()
cutoff = max_date - pd.Timedelta(days=int(time_window_days))
reviews_window = reviews_enriched[reviews_enriched["review_date"] >= cutoff].copy()

# Compute station metrics
stations_view = compute_station_metrics(stations, reviews_window)

# Apply station-level filters
filtered = stations_view.copy()
if borough_filter != "All":
    filtered = filtered[filtered["borough"] == borough_filter]

filtered = filtered[
    (filtered["review_count"] >= min_reviews) &
    (filtered["avg_rating"] >= rating_range[0]) &
    (filtered["avg_rating"] <= rating_range[1])
].copy()

st.write(f"Showing **{len(filtered)}** stations")

# Station details
st.subheader("Station details")

if filtered.empty:
    st.info("No stations match the current filters. Try widening the filters.")
    selected_station_id = None
else:
    station_options = (
        filtered.sort_values(["avg_rating", "review_count"], ascending=[False, False])
        .assign(label=lambda d: d["name"] + " — " + d["address"])
    )
    selected_label = st.selectbox("Select a station", station_options["label"].tolist(), index=0)
    selected_station_id = station_options.loc[
        station_options["label"] == selected_label, "station_id"
    ].iloc[0]

if selected_station_id:
    station_row = stations_view[stations_view["station_id"] == selected_station_id].iloc[0]
    station_reviews = reviews_window[reviews_window["station_id"] == selected_station_id].copy()

    # Key themes
    all_themes = []
    for themes in station_reviews["themes"].tolist():
        all_themes.extend(themes)
    theme_counts = Counter(all_themes)
    top_themes = [t for t, _ in theme_counts.most_common(5)]

    pos = station_reviews[station_reviews["rating"] >= 4].sort_values("review_date", ascending=False).head(3)
    neg = station_reviews[station_reviews["rating"] <= 2].sort_values("review_date", ascending=False).head(3)

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(f"**{station_row['name']}**")
        st.caption(station_row["address"])
        st.metric("Avg rating", station_row["avg_rating_display"])
        st.metric("Reviews (window)", station_row["review_count"])
        st.metric("Negative %", station_row["neg_pct_display"])
        st.write(f"Sentiment: ✅ {station_row['pos_count']}  •  😐 {station_row['neu_count']}  •  ❌ {station_row['neg_count']}")
        st.write("**Key themes (window):** " + (", ".join(top_themes) if top_themes else "None detected"))

    with right:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Top positives")
            if pos.empty:
                st.caption("No 4–5 star reviews in this window.")
            else:
                for _, r in pos.iterrows():
                    st.write(f"⭐ {int(r['rating'])} — {r['review_date'].date()}")
                    st.write(f"“{r['review_text']}”")
                    st.divider()

        with c2:
            st.markdown("### Top negatives")
            if neg.empty:
                st.caption("No 1–2 star reviews in this window.")
            else:
                for _, r in neg.iterrows():
                    st.write(f"⭐ {int(r['rating'])} — {r['review_date'].date()}")
                    st.write(f"“{r['review_text']}”")
                    st.divider()

# Map
st.subheader("Map")
st.caption(f"Time window: last {time_window_days} days (based on latest review date)")

view_state = pdk.ViewState(latitude=51.5072, longitude=-0.1276, zoom=10)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position=["lon", "lat"],
    get_radius=8,
    radius_units="pixels",
    pickable=True,
    auto_highlight=True,
)

tooltip = {
    "text": (
        "Station: {name}\n"
        "Avg rating: {avg_rating_display}\n"
        "Reviews: {review_count_display}\n"
        "Negative %: {neg_pct_display}\n"
        "Borough: {borough}"
    )
}

CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
deck = pdk.Deck(map_style=CARTO_POSITRON, initial_view_state=view_state, layers=[layer], tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)

# Table
st.subheader("Station summary")
st.dataframe(filtered[["name", "borough", "avg_rating", "review_count", "neg_pct_display"]], use_container_width=True)
