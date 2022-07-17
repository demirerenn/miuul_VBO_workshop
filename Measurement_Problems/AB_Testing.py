import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Görev 1: Veriyi Hazırlama ve Analiz Etme

"""Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları 
gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır. Kontrol ve Test grubu olmak üzere iki ayrı 
veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır. Kontrol grubuna 
Maximum Bidding, test grubuna Average Bidding uygulanmıştır. 

4 Değişken 40 Gözlem 26 KB

Impression : Reklam görüntüleme sayısı
Click : Görüntülenen reklama tıklama sayısı
Purchase : Tıklanan reklamlar sonrası satın alınan ürün sayısı
Earning : Satın alınan ürünler sonrası elde edilen kazanç

"""

# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test
# grubu verilerini ayrı değişkenlere atayınız.
control_df_ = pd.read_excel(
    "C:\\Users\\erend\\PycharmProjects\\pythonProject\dsmlbc_9_simge_ilgim\Moduls\\3_Olcumleme_Problemleri\\Datasets"
    "\\ab_testing.xlsx", sheet_name="Control Group")
test_df_ = pd.read_excel(
    "C:\\Users\\erend\\PycharmProjects\\pythonProject\dsmlbc_9_simge_ilgim\Moduls\\3_Olcumleme_Problemleri\\Datasets"
    "\\ab_testing.xlsx", sheet_name="Test Group")

control_df = control_df_.copy()
test_df = test_df_.copy()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

control_df.info()
control_df.describe().T
test_df.info()
test_df.describe().T

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
df = pd.concat([control_df, test_df])
df = df.reset_index()

# Görev 2: A/B Testinin Hipotezinin Tanımlanması

# Adım 1: Hipotezi tanımlayınız.

# H0 : M1 = M2 (Kontrol ve test gruplarında, tıklanan reklamlar sonrası alınan ürün sayısı ortalamarı arasında fark
# yoktur.) H1 : M1!= M2 (Kontrol ve test gruplarında, tıklanan reklamlar sonrası alınan ürün sayısı ortalamarı
# arasında fark vardır.)

# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.

control_mean = df.loc[:39, "Purchase"].mean()
test_mean = df.loc[40:, "Purchase"].mean()

# Kontrol grubu ile test grubunun reklam tıklaması sonra satılan ürün sayısı ortalamalarında fark görünüyor.
# Test grubunda ortalama satılan ürün sayısı daha fazla
# Eldeki ortalama satış sayılarına bakarak, yöntem değişikliğinin işe yarayacağı söylenebilir.

# Görev 3: Hipotez Testinin Gerçekleştirilmesi

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.

# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup
# uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz. Normallik Varsayımı : H0: Normal dağılım varsayımı
# sağlanmaktadır. H1: Normal dağılım varsayımı sağlanmamaktadır. p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ Test
# sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini
# yorumlayınız.

test_stat_control_n, p_value_control_n = shapiro(df.loc[:39, "Purchase"])
test_stat_test_n, p_value_test_n = shapiro(df.loc[40:, "Purchase"])
print("Test Stat Kontrol = %.4f, p-value Kontrol = %.4f \nTest Stat Test = %.4f, p-value Test = %.4f" % (
    test_stat_control_n, p_value_control_n, test_stat_test_n, p_value_test_n))

# Test Stat Kontrol = 0.9773, p-value Kontrol = 0.5891
# Test Stat Test = 0.9589, p-value Test = 0.1541
# p-value'ler 0.05'ten büyük olduğu için varsayım sağlanmaktadır.


# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat_v, p_value_v = levene(df.loc[:39, "Purchase"], df.loc[40:, "Purchase"])
print("Test Stat = %.4f, p-value  = %.4f \n" % (test_stat_v, p_value_v))

# Test Stat = 2.6393, p-value  = 0.1083
# p-value  = 0.1083  > 0.05 olduğu için varsayım homojenliği sağlanmaktadır.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

# Vasayımlar sağlandığı için bağımsız iki öreneklem t testi (parametrik test) yapılacaktır.

test_stat_h, p_value_h = ttest_ind(df.loc[:39, "Purchase"], df.loc[40:, "Purchase"], equal_var=True)
print("Test Stat = %.4f, p-value  = %.4f \n" % (test_stat_h, p_value_h))

# Test Stat = -0.9416, p-value  = 0.3493
# p-value  = 0.3493 > 0.05 olduğu için H0 hipotezi reddedilemez.

# Adım 3: Test sonucunda elde edilen p-value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# Test sonucu elde edilen p_value değeri göz önüne alındığında, kontrol ve test grubu satın alma ortalamaları
# arasında istatistiki olarak anlamlı bir fark olmadığı görülmüştür.


# Görev 4: Sonuçların Analizi

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# Varsayım kontrolünde, normallik ve varyans homojenliği varsayımları sağlandığı için "Bağımsız İki Örneklem T Testi
# (parametrik test)" kullanılmıştır.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
# Maximum Bidding ve  AverageBidding yöntemleri arasında anlamlı bir fark görülmediği için, Maximum Bidding yöntemine
# devam edilmesinde bir sakınca  yoktur. Yeni yöntem için yapılan yatırım gereksiz görülmektedir.