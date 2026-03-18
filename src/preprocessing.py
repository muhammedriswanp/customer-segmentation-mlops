def remove_duplicates(df):
    return df.drop_duplicates()


def map_education_groups(df):
    edu_map = {
    'Basic': 'Undergraduate',
    '2n Cycle': 'Undergraduate',
    'Graduation': 'Graduate',
    'Master': 'Postgraduate',
    'PhD': 'Postgraduate'
    }

    df['Education_Group'] = df['Education'].replace(edu_map)
    df = df.drop(columns=['Education'])
    return df

def clean_marital_status(df):
    df['Marital_Status'] = df['Marital_Status'].replace('Alone','Single')
    df = df[ ~ df['Marital_Status'].isin(['Absurd', 'YOLO'])]
    df['Marital_Status'] = df['Marital_Status'].replace({
    "Married": "Partnered",
    "Together": "Partnered"
    })
    return df

def remove_outliers_income(df):
    df = df[df["Income"] < 600000]
    return df

def handle_missing_values(df):
    df = df.dropna(subset=['Income'])
    return df

def drop_complain(df):
    df = df.drop(columns=['Complain'], errors='ignore')
    return df

def drop_irrelevant_columns(df):
    df = df.drop(columns=['Z_CostContact', 'Z_Revenue', 'ID', 'Response'], errors='ignore')
    return df



def preprocess(df):
    df = remove_duplicates(df)
    df = map_education_groups(df)
    df = clean_marital_status(df)
    df = remove_outliers_income(df)
    df = handle_missing_values(df)
    df = drop_complain(df)
    df = drop_irrelevant_columns(df)
    return df