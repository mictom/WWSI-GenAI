import streamlit as st
import pandas as pd
import json
import spacy
import io
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Setup ---
load_dotenv()
nlp = spacy.load("en_core_web_sm")
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# --- Prompty ---
sentiment_parser = JsonOutputParser()

sentiment_prompt = PromptTemplate(
    template="""
You are a sentiment analysis expert.

Review the following customer review and satisfaction score.
Decide whether the overall sentiment is positive or negative.

Review: ```{review}```
Customer Satisfaction Score (1–5): {score}

Return a valid JSON object in this format:
{{
  "positive_sentiment": boolean,
  "reasoning": string
}}

Make sure your response is valid JSON.
""",
    input_variables=["review", "score"]
)
sentiment_chain = sentiment_prompt | llm | sentiment_parser

negative_prompt = PromptTemplate(
    template="""You are a customer service representative for a company.
A customer has left a negative review about our products or services.

Customer Review: {review}

Based on the review, identify what they specifically disliked and create a personalized response that:
1. Apologizes for their negative experience
2. Addresses the specific issue they mentioned
3. Explains how you'll mitigate this issue in the future
4. Offers a 13% discount on their next visit
5. Thanks them for their feedback

Return your response as a valid JSON object with the following format:
{{"message": str}}

Make sure to add appropriate newline after approximately every 80 characters for better readability.
""",
    input_variables=["review"]
)
negative_chain = negative_prompt | llm | sentiment_parser

positive_prompt = PromptTemplate(
    template="""You are a customer service representative for a company.
A customer has left a positive review about our products or services.

Customer Review: {review}

Based on the review, identify what they specifically liked and create a personalized, short response message that:
1. Thanks them for their positive feedback
2. Offers a voucher related to the thing they liked
3. Encourages them to visit again

Return your response as a valid JSON object with the following format:
{{"message": str}}

Make sure to add appropriate newline after approximately every 80 characters for better readability.
""",
    input_variables=["review"]
)
positive_chain = positive_prompt | llm | sentiment_parser

# --- Common functions ---
def extract_locations(text: str):
    doc = nlp(text)
    return set(ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"])

def recommend_trips_with_ner(review_text: str, trips_df: pd.DataFrame):
    locations = extract_locations(review_text)
    trips_df["score"] = (
        trips_df["Count of days"] * trips_df["Extra activities"].apply(len)
        + trips_df["Cost in EUR"] / 100
    )

    if not locations:
        filtered = trips_df.copy()
    else:
        filtered = trips_df[
            trips_df["City"].isin(locations) | trips_df["Country"].isin(locations)
        ]

    if len(filtered) < 3:
        needed = 3 - len(filtered)
        remaining = trips_df[~trips_df.index.isin(filtered.index)]
        filler = remaining.sort_values(by="score", ascending=False).head(needed)
        final = pd.concat([filtered, filler])
    else:
        final = filtered.sort_values(by="score", ascending=False).head(3)

    selected_cols = [
        "Country", "City", "Start date", "Count of days",
        "Cost in EUR", "Extra activities", "Trip details"
    ]
    return final[selected_cols].to_dict(orient="records")

def calc_metrics(results_df):
    filtered = results_df[results_df["predicted_sentiment"] != "error"]
    accuracy = accuracy_score(filtered["true_sentiment"], filtered["predicted_sentiment"])
    precision = precision_score(filtered["true_sentiment"], filtered["predicted_sentiment"], pos_label="positive")
    recall = recall_score(filtered["true_sentiment"], filtered["predicted_sentiment"], pos_label="positive")
    return accuracy, precision, recall

# === Page Selection Interface ===
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio("Choose a page:", ["🖼️ System diagram", "📊 Batch Sentiment Analysis", "💬 Individual Review Assistant"])

# === Shared data upload ===
st.sidebar.header("📂 Upload Data")
reviews_file = st.sidebar.file_uploader("Upload Customer Reviews (JSON)", type="json")
trips_file = st.sidebar.file_uploader("Upload Trips Data (JSON)", type="json")

# === Loading trip data ===
trips_df = pd.DataFrame()
if trips_file is not None:
    try:
        trips = json.load(io.StringIO(trips_file.getvalue().decode("utf-8")))
        trips_df = pd.DataFrame(trips)
    except json.JSONDecodeError as e:
        st.error(f"Error loading uploaded trips JSON: {e}")
else:
    try:
        with open("./trips_data.json", "r", encoding="utf-8") as f:
            trips = json.load(f)
            trips_df = pd.DataFrame(trips)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Default trips file could not be loaded: {e}")

# === Page 1: System Diagram ===
if page == "🖼️ System diagram":
    st.title("🖼️ Sentiment Analysis + Travel Recommendations")

    st.markdown("Below is the diagram of the system architecture:")

    try:
        st.image("./Schemat_systemu.png", caption="System Architecture")
    except FileNotFoundError:
        st.error("The file 'Schemat_systemu.png' was not found. Please make sure it's in the same folder as the script.")

# === Page 2: Aggregate Review Analysis ===
elif page == "📊 Batch Sentiment Analysis":
    st.title("📊 Sentiment Analysis + Travel Recommendations")
    if reviews_file and not trips_df.empty:
        df = pd.read_json(reviews_file)

        st.success("Files loaded successfully!")

        results = []
        with st.spinner("🔍 Analyzing sentiment..."):
            for row in df.itertuples(index=False):
                try:
                    response = sentiment_chain.invoke({
                        "review": row.review,
                        "score": row.customer_satisfaction_score
                    })
                    predicted = "positive" if response["positive_sentiment"] else "negative"
                    reasoning = response["reasoning"]
                except Exception as e:
                    predicted = "error"
                    reasoning = str(e)
                results.append({
                    "review": row.review,
                    "true_sentiment": row.survey_sentiment,
                    "predicted_sentiment": predicted,
                    "reasoning": reasoning,
                    "score": row.customer_satisfaction_score
                })

        results_df = pd.DataFrame(results)
        acc, prec, rec = calc_metrics(results_df)

        st.subheader("📈 Classification Metrics")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**Precision:** {prec:.2f}")
        st.write(f"**Recall:** {rec:.2f}")

        result_column = []
        for row in results_df.itertuples(index=False):
            if row.predicted_sentiment == "negative":
                result_column.append("13% discount")
            elif row.predicted_sentiment == "positive":
                recs = recommend_trips_with_ner(row.review, trips_df)
                result_column.append(", ".join([trip["City"] for trip in recs]))
            else:
                result_column.append("unable to classify")

        results_df["recommendation/discount"] = result_column

        st.subheader("🔍 Detailed Results")
        st.dataframe(results_df[["review", "true_sentiment", "predicted_sentiment", "reasoning", "recommendation/discount"]])

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, file_name="sentiment_results.csv", mime="text/csv")
    else:
        st.warning("Please upload both the reviews and trips data to proceed.")

# === Page 3: Interactive Review Assistant ===
elif page == "💬 Individual Review Assistant":
    st.title("💬 Individual Review Assistant")
    user_review = st.text_area("Enter customer review text:")
    user_score = st.slider("Customer Satisfaction Score", min_value=1, max_value=5, value=4)

    if st.button("🧠 Analyze and Generate Response"):
        if user_review.strip():
            sentiment_response = sentiment_chain.invoke({
                "review": user_review,
                "score": user_score
            })

            is_positive = sentiment_response["positive_sentiment"]
            reasoning = sentiment_response["reasoning"]

            st.subheader("🧪 Sentiment Classification")
            st.write("**Sentiment:**", "😊 Positive" if is_positive else "😠 Negative")
            st.write("**Reasoning:**", reasoning)

            if is_positive:
                response = positive_chain.invoke({"review": user_review})
                recs = recommend_trips_with_ner(user_review, trips_df)

                st.subheader("💌 GPT-generated Response to Customer")
                st.text_area("Response", response["message"], height=150)

                st.subheader("🌍 Recommended Trips")
                for idx, trip in enumerate(recs, 1):
                    with st.expander(f"🧳 Trip {idx}: {trip['Country']} - {trip['City']} ({trip['Start date']})"):
                        st.write(f"**Country:** {trip['Country']}")
                        st.write(f"**City:** {trip['City']}")
                        st.write(f"**Start date:** {trip['Start date']}")
                        st.write(f"**Duration:** {trip['Count of days']} days")
                        st.write(f"**Cost:** €{trip['Cost in EUR']}")
                        st.write(f"**Extra activities:** {', '.join(trip['Extra activities'])}")
                        st.write(f"**Trip details:** {trip['Trip details']}")
            else:
                response = negative_chain.invoke({"review": user_review})
                st.subheader("🙏 GPT-generated Apology Message")
                st.text_area("Apology", response["message"], height=150)
                st.write("💸 Offer: **13% discount** for next visit")
        else:
            st.warning("Please enter some review text before clicking analyze.")
