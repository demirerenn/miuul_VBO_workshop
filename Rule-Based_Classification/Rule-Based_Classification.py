import pandas as pd

"Reading persona.csv and showing general information about the dataset"
df = pd.read_csv(
    "C:\\Users\\erend\\PycharmProjects\\pythonProject\\dsmlbc_9_simge_ilgim\\Homeworks\\ErenDemir\\Week2_PythonileVeriAnalizi\\persona.csv")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()

"unique SOURCE number and frequency information"
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

"number of unique PRICE"
df["PRICE"].nunique()

"Sales information by PRICE type"
df["PRICE"].value_counts()

"number of sales by country"
df["COUNTRY"].value_counts()

"Total earnings by COUNTRY""
df.pivot_table("PRICE", "COUNTRY", aggfunc="sum")

"Number of sales by SOURCE variable"
df["SOURCE"].value_counts()

"PRICE averages by COUNTRY"
df.pivot_table("PRICE", "COUNTRY", aggfunc="mean")

"PRICE averages according to the SOURCE variable"
df.pivot_table("PRICE", "SOURCE", aggfunc="mean")

"PRICE averages in the COUNTRY-SOURCE breakdown"
df.pivot_table("PRICE", "COUNTRY", "SOURCE", aggfunc="mean")

"Average earnings in COUNTRY, SOURCE, SEX, AGE breakdown"
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

"alignment"
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False)

"convert index to variable"
agg_df = agg_df.reset_index()

"Converting AGE variable to categorical variable and adding to agg_df"
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, 70], labels=["14_18", "19_23", "24_30", "31_40", "41_66"])

"Level-based customers"
agg_df['customers_level_based'] = ["_".join(dict.fromkeys(v)).upper() for v in
                                   agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].values]

agg_df[["customers_level_based", "PRICE"]]

agg_df_final = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df_final = agg_df_final.reset_index()

"customer segmentation"

agg_df_final['SEGMENT'] = pd.qcut(agg_df_final['PRICE'], q=4, labels=['D', 'C', 'B', 'A'])
agg_df_final.groupby("SEGMENT").agg({"PRICE": ["count", "mean", "sum", "max", "min"]}).reset_index()

"new customer segmentation"

print(agg_df_final[agg_df_final["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"])
print(agg_df_final[agg_df_final["customers_level_based"] == "FRA_IOS_FEMALE_31_40"])
