import datetime as dt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



# reading dataset and making a copy of dataframe
df_ = pd.read_csv(
    "C:\\Users\\erend\\PycharmProjects\\pythonProject\\dsmlbc_9_simge_ilgim\\Moduls\\2_CRM_Analitigi\Datasets"
    "\\flo_data_20k.csv")
df = df_.copy()

# top 10 observations
df.head(10)

# variable names
df.columns

# descriptive statistics for numerical variables
df.describe().T

# number of blank observations according to variables
df.isnull().sum()

# variable data types
df.dtypes

# total number of purchases and total spend of each customer

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Converting the type of variables that express dates

# df[["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]] = df[
#    ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]].apply(pd.to_datetime)

# df[["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]] = df[
#    ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]].astype(
#    'datetime64[D]')

df.dtypes
df.head()
df.columns

columns_list = df.columns
date_list = []
for i in columns_list:
    if "date" in i:
        date_list.append(i)
        df[date_list] = df[date_list].apply(pd.to_datetime)

df.dtypes


df.groupby("order_channel").agg({"master_id": "count", "order_num_total": "sum", "customer_value_total": "sum"})

# top 10 customers with the most revenue

df.sort_values(by="customer_value_total", ascending=False).head(10)

# top 10 customers with the most orders

df.sort_values(by="order_num_total", ascending=False).head(10)



def data_preparation(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + \
                                        dataframe["customer_value_total_ever_offline"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    return dataframe


# data_preparation(df)

# RFM

#  Recency, Frequency and Monetary

today_date = dt.datetime(2021, 6, 1)
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()
rfm.info()

# Calculation of RF Score

# Converting Recency, Frequency and Monetary metrics to scores between 1-5 with the help of qcut

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5,
                                 labels=[1, 2, 3, 4, 5])  # rank frekansların çakışmasını engeller
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# Definition of RF Score as Segments

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
# rfm.groupby("segment")[["segment", "recency", "frequency", "monetary"]].agg(["mean", "count"])

# RFM for a customer

rfm = rfm.reset_index()
df_final = pd.merge(df, rfm)

rfm_offer_list = df_final[
    ((df_final["segment"] == "champions") | (df_final["segment"] == "loyal_customers")) & df_final[
        "interested_in_categories_12"].str.contains("KADIN", na=False)].index

offer_list = df_final.loc[rfm_offer_list, :].master_id
offer_list.to_csv("offer_list.csv", index=False)

# RFM for a customer

rfm_offer_list2 = df_final[((df_final["segment"] == "about_to_sleep") | (df_final["segment"] == "cant_loose") | \
                            (df_final["segment"] == "new_customers")) & df_final["interested_in_categories_12"]. \
                               str.contains("ERKEK", "COCUK", na=False)].index
offer_list2 = df_final.loc[rfm_offer_list2, :].master_id
offer_list2.to_csv("offer_list2.csv", index=False)
