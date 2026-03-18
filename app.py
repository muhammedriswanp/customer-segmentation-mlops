import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load saved models ──────────────────────────
scaler = joblib.load('outputs/models/scaler.pkl')
pca    = joblib.load('outputs/models/pca.pkl')
model  = joblib.load('outputs/models/kmeans_model.pkl')
df     = pd.read_csv('outputs/clusters/dataset_with_clusters.csv')

# ── Cluster info ───────────────────────────────
cluster_info = {
    0: {
        "name": "🟣 Budget Conscious Families",
        "description": "Low income, high number of children, price-sensitive customers who browse frequently but spend minimally.",
        "recommendation": "Target with discount deals, family bundles, and budget-friendly promotions."
    },
    1: {
        "name": "🟡 High Value Loyalists",
        "description": "Highest income, maximum spending, campaign-responsive, prefer catalogue shopping.",
        "recommendation": "Target with premium products, exclusive catalogues, and personalized campaigns."
    },
    2: {
        "name": "🔵 Middle Class Actives",
        "description": "Mid-range income, loyal customers with moderate spending across all channels.",
        "recommendation": "Reward loyalty with membership programs and mid-range product promotions."
    }
}

cluster_colors = ['#8e44ad', '#f1c40f', '#2980b9']
cluster_names  = {k: v["name"] for k, v in cluster_info.items()}

# ── Navigation ─────────────────────────────────
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Cluster Overview",
    "🔬 Feature Analysis",
    "🔍 Predict Segment"
])

# ══════════════════════════════════════════════
# PAGE 1 — Cluster Overview
# ══════════════════════════════════════════════
if page == "📊 Cluster Overview":
    st.title("📊 Cluster Overview")
    st.markdown("Summary of the 3 customer segments identified by KMeans clustering.")

    # ── Cluster sizes ──
    st.subheader("Cluster Sizes")
    counts = df['KMeans_Cluster'].value_counts().sort_index()
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar([cluster_names[i] for i in counts.index],
               counts.values, color=cluster_colors)
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customers per Cluster")
        plt.xticks(rotation=15, ha='right')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(counts.values,
               labels=[cluster_names[i] for i in counts.index],
               colors=cluster_colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("Cluster Distribution")
        st.pyplot(fig)

    # ── Key metrics ──
    st.subheader("Key Metrics per Cluster")
    profile = df.groupby('KMeans_Cluster').agg({
        'Income': 'mean',
        'Total_Spending': 'mean',
        'Total_Children': 'mean',
        'Total_Campaign_Accepted': 'mean',
        'Spending_Per_Purchase': 'mean'
    }).round(2)
    profile.index = [cluster_names[i] for i in profile.index]
    st.dataframe(profile)

# ══════════════════════════════════════════════
# PAGE 2 — Feature Analysis
# ══════════════════════════════════════════════
elif page == "🔬 Feature Analysis":
    st.title("🔬 Feature Analysis")
    st.markdown("Compare feature distributions across customer segments.")

    feature = st.selectbox("Select Feature", [
        'Income', 'Total_Spending', 'Total_Purchases',
        'Total_Children', 'Total_Campaign_Accepted',
        'NumWebVisitsMonth', 'Spending_Per_Purchase'
    ])

    clean_names = {0: "Budget Conscious Families", 
                   1: "High Value Loyalists", 
                   2: "Middle Class Actives"}

    col1, col2 = st.columns(2)

    # ── Histogram ──
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for cluster_id, color in zip([0, 1, 2], cluster_colors):
            subset = df[df['KMeans_Cluster'] == cluster_id][feature]
            sns.kdeplot(subset, ax=ax, color=color, 
                        label=clean_names[cluster_id], 
                        fill=True, alpha=0.4, linewidth=2)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {feature}")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)


    # ── Box plot ──
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        data_per_cluster = [df[df['KMeans_Cluster'] == i][feature].values for i in [0, 1, 2]]
        bp = ax.boxplot(data_per_cluster, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], cluster_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(["Budget\nFamilies", "High Value\nLoyalists", "Middle Class\nActives"])
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} by Cluster")
        st.pyplot(fig)

    # ── Average bar chart ──
    st.subheader(f"Average {feature} per Cluster")
    means = df.groupby('KMeans_Cluster')[feature].mean().round(2)
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    bars = ax3.bar(
        ["Budget Conscious\nFamilies", "High Value\nLoyalists", "Middle Class\nActives"],
        means.values, color=cluster_colors, edgecolor='white', linewidth=0.5
    )
    for bar, val in zip(bars, means.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                 f'{val:,.1f}', ha='center', fontsize=10, fontweight='bold')
    ax3.set_ylabel(feature)
    ax3.set_title(f"Average {feature} per Cluster")
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    st.pyplot(fig3)


# ══════════════════════════════════════════════
# PAGE 3 — Predict Segment
# ══════════════════════════════════════════════
elif page == "🔍 Predict Segment":
    st.title("🛍️ Customer Segment Predictor")
    st.markdown("Enter customer details below to predict which segment they belong to.")

    col1, col2, col3 = st.columns(3)

    with col1:
        income              = st.number_input("Annual Income ($)",              1000, 200000, 50000)
        recency             = st.number_input("Days Since Last Purchase",       0, 100, 30)
        total_children      = st.number_input("Number of Children",             0, 5, 1)
        age                 = st.number_input("Age",                            18, 100, 40)
        tenure_days         = st.number_input("Customer Tenure (Days)",         0, 1000, 300)

    with col2:
        mnt_wines           = st.number_input("Spent on Wines - last 2 years ($)",   0, 1500, 100)
        mnt_fruits          = st.number_input("Spent on Fruits - last 2 years ($)",  0, 200, 20)
        mnt_meat            = st.number_input("Spent on Meat - last 2 years ($)",    0, 1500, 100)
        mnt_fish            = st.number_input("Spent on Fish - last 2 years ($)",    0, 300, 10)
        mnt_sweet           = st.number_input("Spent on Sweets - last 2 years ($)",  0, 300, 10)
        mnt_gold            = st.number_input("Spent on Gold - last 2 years ($)",    0, 400, 50)

    with col3:
        num_web_purchases   = st.number_input("Web Purchases - last 2 years",   0, 30, 5)
        num_store_purchases = st.number_input("Store Purchases - last 2 years", 0, 20, 5)
        num_catalog         = st.number_input("Catalog Purchases - last 2 years",0, 30, 2)
        num_deals           = st.number_input("Deal Purchases - last 2 years",  0, 15, 1)
        num_web_visits      = st.number_input("Web Visits/Month",               0, 20, 5)
        campaigns_accepted  = st.number_input("Campaigns Accepted (0-6)",       0, 6, 0)

    if st.button("🔍 Predict My Segment"):

        total_spending        = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
        total_purchases       = num_web_purchases + num_store_purchases + num_catalog + num_deals
        spending_per_purchase = total_spending / (total_purchases + 1)

        mnt_wines_log  = np.log1p(mnt_wines)
        mnt_fruits_log = np.log1p(mnt_fruits)
        mnt_meat_log   = np.log1p(mnt_meat)
        mnt_fish_log   = np.log1p(mnt_fish)
        mnt_sweet_log  = np.log1p(mnt_sweet)
        mnt_gold_log   = np.log1p(mnt_gold)

        input_data = pd.DataFrame([[
            income, recency, mnt_wines_log, mnt_fruits_log, mnt_meat_log,
            mnt_fish_log, mnt_sweet_log, mnt_gold_log,
            num_deals, num_web_purchases, num_catalog, num_store_purchases,
            num_web_visits, age, total_children, campaigns_accepted,
            tenure_days, total_purchases, total_spending, spending_per_purchase,
            1, 0, 0, 0, 0
        ]], columns=[
            'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'Total_Children',
            'Total_Campaign_Accepted', 'Customer_Tenure_Days', 'Total_Purchases',
            'Total_Spending', 'Spending_Per_Purchase',
            'Marital_Status_Partnered', 'Marital_Status_Single',
            'Marital_Status_Widow', 'Education_Group_Postgraduate',
            'Education_Group_Undergraduate'
        ])

        input_scaled    = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
        input_pca       = pca.transform(input_scaled_df)
        cluster         = model.predict(input_pca)[0]

        info = cluster_info[cluster]
        st.success(f"### Predicted Segment: {info['name']}")
        st.info(f"**Profile:** {info['description']}")
        st.warning(f"**Recommendation:** {info['recommendation']}")
        st.markdown(f"**Debug — Cluster ID:** `{cluster}`")