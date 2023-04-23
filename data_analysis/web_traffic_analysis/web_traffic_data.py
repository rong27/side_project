import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlsplit, parse_qs # 內建網址解析套件

connection = pymysql.connect(host = 'localhost',
                             user = 'root',
                             password = '',
                             port = 3307,
                             db = 'demo_shop_logs',
                             charset = 'utf8mb4',
                             cursorclass = pymysql.cursors.DictCursor) # cursorclass 為使用 dict 取代 tuple 當作回傳資料格式


# Table: user_purchase_logs

try:
    with connection.cursor() as cursor:
        sql = 'SELECT * FROM user_purchase_logs'
        cursor.execute(sql)
        items = cursor.fetchall()
        # print(items)

finally:
    connection.close()


# 建立 dict 儲存 UTM 統計資料
utm_stats = {
    'utm_source': {},
    'utm_medium': {},
    'utm_campaign': {}
}

# 將 SQL 查詢的資料一一取出
for item in items:
    referrer = item['referrer']
    # 透過 urlsplit 將 referrer 網址分解成網址物件,取出屬性 query
    query_str = urlsplit(referrer).query
    # 將網址後面所接的參數轉為 dict：{}
    query_dict = parse_qs(query_str)

    # 將 query 一一取出
    for query_key, query_value in query_dict.items():
        utm_value = query_value[0]
        if utm_value in utm_stats[query_key]:
            utm_stats[query_key][utm_value] += 1
        else:
            utm_stats[query_key][utm_value] = 1

# 將資料轉為 pandas Series
df_utm_source = pd.Series(utm_stats['utm_source'])

df_utm_medium = pd.Series(utm_stats['utm_medium'])

df_utm_campaign = pd.Series(utm_stats['utm_campaign'])


# 若建立多個子圖表 subplots 於同一個畫面中，可以使用 subplots
# nrows 代表列，代表 ncols 行
fig, axes = plt.subplots(nrows=2, ncols=3)

df_utm_source.plot(ax=axes[0, 0], kind='bar', rot=0)
axes[0, 0].set_title('utm source')

df_utm_medium.plot(ax=axes[0, 1], kind='bar', rot=0)
axes[0, 1].set_title('utm medium')

df_utm_campaign.plot(ax=axes[0, 2], kind='bar', rot=0)
axes[0, 2].set_title('utm campaign')

for ax in axes.flat:
    for i, v in enumerate(ax.patches):    # 可以透過 Axes 物件中的 patches 屬性來遍歷每個子圖中的所有物件
        x = v.get_x() + v.get_width()/2   # v.get_x()     矩形的左邊緣的 x 座標值
        y = v.get_height()                # v.get_width() 形的寬度
        ax.annotate(str(y), (x, y), ha='center', va='bottom')

axes[1, 0].pie(df_utm_source.values, labels=df_utm_source.index, autopct='%1.1f%%')
axes[1, 0].set_title('utm_source')

axes[1, 1].pie(df_utm_medium.values, labels=df_utm_medium.index, autopct='%1.1f%%')
axes[1, 1].set_title('utm_medium')

axes[1, 2].pie(df_utm_campaign.values, labels=df_utm_campaign.index, autopct='%1.1f%%')
axes[1, 2].set_title('utm_campaign')


plt.show()
