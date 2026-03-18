import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_age(df):
    df['Age'] = 2026 - df['Year_Birth']
    df = df.drop(columns=['Year_Birth'])
    return df

def remove_outliers_age(df):
    df = df[df['Age']  <= 100 ]
    return df

def create_children(df):
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df = df.drop(columns=['Kidhome','Teenhome'])
    return df


def create_campaign_total(df):
    cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                'AcceptedCmp4', 'AcceptedCmp5']
    df['Total_Campaign_Accepted'] = df[cmp_cols].sum(axis=1)
    df = df.drop(columns=cmp_cols)
    return df

def create_tenure(df):
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df = df.drop(columns=['Dt_Customer'])
    return df

def create_purchase_total(df):
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases',
                     'NumStorePurchases', 'NumDealsPurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    return df

def create_spending_total(df):
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spending'] = df[spend_cols].sum(axis=1)
    return df

def create_spending_per_purchase(df):
    df['Spending_Per_Purchase'] = df['Total_Spending'] / (df['Total_Purchases'] + 1)
    return df

def encode_categoricals(df):
    df = pd.get_dummies(df, columns=['Marital_Status', 'Education_Group'],
                        drop_first=True, dtype=int)
    return df

def apply_log_transform(df):
    cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df[cols] = np.log1p(df[cols])
    return df

def scale_features(df):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return X_scaled, scaler


def engineer_features(df):
    df = create_age(df)
    df = remove_outliers_age(df)
    df = create_children(df)
    df = create_campaign_total(df)
    df = create_tenure(df)
    df = create_purchase_total(df)
    df = create_spending_total(df)
    df = create_spending_per_purchase(df)
    df = apply_log_transform(df)
    df = encode_categoricals(df)

    return df
