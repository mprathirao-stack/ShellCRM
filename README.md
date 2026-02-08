# Shell London Reviews — Customer Experience Dashboard (Demo App)

This is a simple interactive dashboard that helps a Customer Experience team quickly understand what customers are saying about Shell gas stations across London, based on Google Reviews.

It is designed for two types of users:
1) **Managers / stakeholders** who want a high-level summary  
2) **Analysts / CX team members** who want to explore individual stations and review evidence

---

## What this app helps you do

### 1) Get an executive summary (overall view)
The app shows:
- Overall customer sentiment (positive / neutral / negative)
- Average rating across stations
- The main reasons people are happy or unhappy (cleanliness, staff, toilets, queues, etc.)
- Whether things are getting better or worse compared to the previous period
- Which stations are improving vs deteriorating

---

### 2) Explore stations on a map (drill down)
The app includes a map of London with Shell stations shown as pins.

You can filter stations by:
- Time period (last 30 / 90 / 365 days)
- Minimum number of reviews
- Average rating range
- Borough (area)

When you select a station, the app shows:
- Average rating and number of reviews
- Sentiment breakdown (how many reviews were positive / neutral / negative)
- Key themes mentioned in that station’s reviews
- Examples of real customer comments (top positives and top negatives)

---

### 3) Ask questions using the chatbot (Q&A)
The app includes a chatbot where you can ask questions like:
- “Which stations have the most complaints about cleanliness?”
- “What are the top reasons for 1-star reviews?”
- “Which stations improved the most in the last 90 days?”
- “Are there recurring mentions of safety concerns?”
- “Summarize feedback about EV charging availability.”

The chatbot responds with:
- Specific station names
- Counts and rankings
- Evidence snippets from real reviews
- A warning if there is not enough information

---

## What makes this useful for a CX team

Instead of reading hundreds of reviews manually, the app:
- Summarizes the overall story in seconds
- Highlights recurring issues
- Helps spot trends (what is getting worse or better)
- Provides supporting review examples to share with operations teams

---

## What data the app uses

The app uses two simple datasets:

### 1) Station list
Each station includes:
- Station name
- Location (latitude/longitude)
- Address
- Borough

### 2) Review list
Each review includes:
- Station ID
- Star rating (1–5)
- Review text
- Review date

---

## How sentiment is calculated (simple explanation)

The app reads the review text and classifies it as:
- **Positive**
- **Neutral**
- **Negative**

This is done automatically using a standard sentiment model (VADER), commonly used for short customer feedback.

---

## How themes are detected (simple explanation)

The app looks for common words related to key experience topics, such as:
- Cleanliness
- Toilets
- Staff behavior
- Queues and waiting time
- Safety
- EV charging
- Car wash

A review can mention more than one theme.

---

## Pages in the app

### 1) Home (Executive Summary)
A management-friendly summary of sentiment, drivers, and trends.

### 2) Map Explorer
Interactive station exploration with filters, station drilldown, and evidence.

### 3) Chatbot
A Q&A interface for quick investigation with evidence-based answers.

---

## Notes / Limitations (important)

This is a demo-ready app designed to be simple and explainable.

- Themes are detected using keyword matching (not deep AI topic modeling)
- The chatbot is rule-based (not a full AI assistant)
- Results depend on the volume and quality of review data available

---

## Future enhancements (optional)

If this were extended into a production system, it could include:
- Smarter AI topic modeling
- LLM-based chatbot (ChatGPT / Azure OpenAI)
- Alerts for sudden spikes in complaints
- Better trend analysis across weeks/months
- Automatic station clustering by area

---

## Deployment
The app can be deployed as a shareable web link using Streamlit Cloud.

---

## Summary
This project provides a fast, practical way to understand customer sentiment and operational issues across Shell stations in London, using real review evidence and easy exploration tools.

App URL: https://shellcrm-h8zuarutxfnupkhdkwb2dr.streamlit.app/
