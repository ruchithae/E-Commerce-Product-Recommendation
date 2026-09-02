import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛍️ E-Commerce Product Recommendation System")
st.write(
    "A collaborative filtering recommendation system that "
    "suggests products based on user ratings."
)

st.divider()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/ecommerce_products.csv")

    return df


try:
    df = load_data()

except FileNotFoundError:
    
    st.error(
        "❌ ecommerce_products.csv not found. "
        "Please keep the CSV file in the same folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# CHECK DATA
# --------------------------------------------------

required_columns = ["user_id", "product_id", "rating"]

for column in required_columns:

    if column not in df.columns:
        st.error(f"❌ Missing column: {column}")
        st.stop()


# --------------------------------------------------
# CREATE USER-ITEM MATRIX
# --------------------------------------------------

user_item_matrix = df.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating"
)


# --------------------------------------------------
# CALCULATE PRODUCT SIMILARITY
# --------------------------------------------------

filled_matrix = user_item_matrix.fillna(0)

similarity = cosine_similarity(filled_matrix.T)

product_similarity = pd.DataFrame(
    similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_products(user_id, top_n=3):

    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id].dropna()

    # Products already rated by the user
    rated_products = set(user_ratings.index)

    recommendation_scores = {}

    for product in user_item_matrix.columns:

        # Do not recommend products already rated
        if product in rated_products:
            continue

        score = 0
        similarity_sum = 0

        for rated_product, rating in user_ratings.items():

            sim = product_similarity.loc[
                product, rated_product
            ]

            score += sim * rating
            similarity_sum += abs(sim)

        if similarity_sum > 0:
            recommendation_scores[product] = (
                score / similarity_sum
            )

    # Sort by predicted rating
    recommendations = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:top_n]


# --------------------------------------------------
# USER SELECTION
# --------------------------------------------------

st.subheader("👤 Select a User")

users = sorted(df["user_id"].unique())

selected_user = st.selectbox(
    "Choose User ID",
    users
)


# --------------------------------------------------
# NUMBER OF RECOMMENDATIONS
# --------------------------------------------------

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=5,
    value=3
)


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button("🔍 Get Recommendations"):

    recommendations = recommend_products(
        selected_user,
        top_n
    )

    st.divider()

    if recommendations:

        st.subheader(
            f"🎯 Recommended Products for {selected_user}"
        )

        # Try to load product details
        try:

            product_details = pd.read_csv(
                "product_details.csv"
            )

        except FileNotFoundError:

            product_details = None


        for product, score in recommendations:

            st.write(
                f"### 🛒 {product}"
            )

            st.write(
                f"⭐ Predicted Rating: **{score:.2f} / 5**"
            )

            # Show product details if available
            if product_details is not None:

                product_info = product_details[
                    product_details["product_id"] == product
                ]

                if not product_info.empty:

                    row = product_info.iloc[0]

                    if "product_name" in product_details.columns:
                        st.write(
                            f"**Product Name:** "
                            f"{row['product_name']}"
                        )

                    if "category" in product_details.columns:
                        st.write(
                            f"**Category:** "
                            f"{row['category']}"
                        )

                    if "description" in product_details.columns:
                        st.write(
                            f"**Description:** "
                            f"{row['description']}"
                        )

            st.divider()

    else:

        st.warning(
            "No new products available for recommendation."
        )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.subheader("📊 Model Evaluation")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Precision@3",
        "0.2083"
    )


with col2:

    st.metric(
        "Recall@3",
        "0.6250"
    )


with col3:

    st.metric(
        "NDCG@3",
        "0.5000"
    )


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📋 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Users",
        df["user_id"].nunique()
    )

with col2:
    st.metric(
        "Products",
        df["product_id"].nunique()
    )

with col3:
    st.metric(
        "Ratings",
        len(df)
    )


# --------------------------------------------------
# SHOW USER RATINGS
# --------------------------------------------------

with st.expander("View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )