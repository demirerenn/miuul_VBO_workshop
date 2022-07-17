import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# reading dataset
df_ = pd.read_csv(
    "C:\\Users\\erend\\PycharmProjects\\pythonProject\\dsmlbc_9_simge_ilgim\\Moduls\\2_CRM_Analitigi\Datasets"
    "\\flo_data_20k.csv")

df = df_.copy()


# Define outlier_thresholds and replace_with_thresholds functions needed to suppress outliers

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# suppressing outliers

df.describe().T

sns.boxplot(x=df["order_num_total_ever_online"])
plt.show()
replace_with_thresholds(df, "order_num_total_ever_online")

sns.boxplot(x=df["order_num_total_ever_offline"])
plt.show()
replace_with_thresholds(df, "order_num_total_ever_offline")

sns.boxplot(x=df["customer_value_total_ever_offline"])
plt.show()
replace_with_thresholds(df, "customer_value_total_ever_offline")

sns.boxplot(x=df["customer_value_total_ever_online"])
plt.show()
replace_with_thresholds(df, "customer_value_total_ever_online")

# total number of purchases and total spend of each customer

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Converting the type of variables that express dates

df.dtypes

columns_list = df.columns
date_list = []
for i in columns_list:
    if "date" in i:
        date_list.append(i)
        df[date_list] = df[date_list].apply(pd.to_datetime)

# date_columns = df.columns[df.columns.str.contains("date")]
# df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Creating the CLTV Data Structure

today_date = dt.datetime(2021, 6, 1)

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
                        "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]).dt.days / 7,
                        "T_weekly": ((today_date - df["first_order_date"]).astype('timedelta64[D]')) / 7,
                        "frequency": df["order_num_total"],
                        "monetary_cltv_avg": df["customer_value_total"] / df["order_num_total"]})

# Establishment of BG/NBD, Gamma-Gamma Models and Calculation of CLTV
# 3 and 6 month purchase forecasts

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
# 3 months
cltv_df["exp_sales_3_month"] = bgf.predict(12, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# 6 months
cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg']).sort_values(
    ascending=False).head(10)

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

cltv_df.sort_values("exp_average_value", ascending=False).head(10)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = pd.merge(cltv_df, cltv, left_index=True, right_index=True)
cltv_final.sort_values(by="clv", ascending=False).head(20)

del cltv_final["index"]

# Customer segmentation by CLTV

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_df.groupby("segment").agg({"frequency": ("mean", "std"),
                                "monetary_cltv_avg": "std"})

cltv_final.groupby("segment").agg(
    {"mean", "std", "median"})
