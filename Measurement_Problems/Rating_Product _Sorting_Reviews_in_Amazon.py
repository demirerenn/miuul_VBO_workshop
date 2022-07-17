import math
import pandas as pd
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

"""
Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir. 
Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

12 Değişken 4915 Gözlem 71.9 MB

reviewerID : Kullanıcı ID’si
asin : Ürün ID’si
reviewerName : Kullanıcı Adı
helpful : Faydalı değerlendirme derecesi
reviewText : Değerlendirme
overall : Ürün rating’i
summary : Değerlendirme özeti
unixReviewTime : Değerlendirme zamanı
reviewTime : Değerlendirme zamanı Raw
day_diff : Değerlendirmeden itibaren geçen gün sayısı
helpful_yes : Değerlendirmenin faydalı bulunma sayısı
total_vote : Değerlendirmeye verilen oy sayısı

"""

# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen
# puanları tarihe göre ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı
# puanın karşılaştırılması gerekmektedir.

# Adım 1: Ürünün ortalama puanını hesaplayınız.

df = pd.read_csv(
    "C:\\Users\\erend\\PycharmProjects\\pythonProject\\dsmlbc_9_simge_ilgim\Moduls\\3_Olcumleme_Problemleri\\Datasets"
    "\\amazon_review.csv")

df.head()
df.info()
df.describe()

print("Ürünün ortalama puanı : %.4f" % df["overall"].mean())

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

# • reviewTime değişkenini tarih değişkeni olarak tanıtmanız • reviewTime'ın max değerini current_date olarak kabul
# etmeniz • her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken
# oluşturmanız ve gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça
# çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir. Örneğin q1 = 12 ise
# ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()
df["days"] = (current_date - df["reviewTime"]).dt.days

df["period"] = pd.qcut(df['days'], 4, labels=[1, 2, 3, 4])

time_based_weighted_average = df.loc[df["period"] == 1, "overall"].mean() * 0.22 + \
                              df.loc[df["period"] == 2, "overall"].mean() * 0.24 + \
                              df.loc[df["period"] == 3, "overall"].mean() * 0.26 + \
                              df.loc[df["period"] == 4, "overall"].mean() * 0.28

print("Ürünün ağırlıklı puan ortalaması : %.4f" % time_based_weighted_average)

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

# ilk çeyrek : 4.6949
# ikinci çeyrek : 4.6371
# üçüncü çeyrek : 4.5717
# son çeyrek : 4.4467
# ilk çeyrekten son çeyreğe doğru puanlar düşmüş. Uzak zamandan yakın zamana doğru ürün memnuniyeti azalmış denebilir.

print("ilk çeyrek : %.4f" % df.loc[df["period"] == 1, "overall"].mean())
print("ikinci çeyrek : %.4f" % df.loc[df["period"] == 2, "overall"].mean())
print("üçüncü çeyrek : %.4f" % df.loc[df["period"] == 3, "overall"].mean())
print("son çeyrek : %.4f" % df.loc[df["period"] == 4, "overall"].mean())

# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.

# Adım 1: helpful_no değişkenini üretiniz. • total_vote bir yoruma verilen toplam up-down sayısıdır. • up,
# helpful demektir. • Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi
# gerekmektedir. • Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan
# oy sayılarını (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
print(df.head())

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

# • score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
# • score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
# • score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# • wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

def score_pos_neg_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

print(df.sort_values("wilson_lower_bound", ascending=False).head(20))