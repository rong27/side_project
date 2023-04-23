import pymysql
import pandas as pd
import matplotlib.pyplot as plt

connection = pymysql.connect(host = 'localhost',
                             user = 'root',
                             password = '',
                             port = 3307,
                             db = 'demo_shop_logs',
                             charset = 'utf8mb4',
                             cursorclass = pymysql.cursors.DictCursor)

try:
    
    with connection.cursor() as cursor:
        sql = """
            SELECT 
                keyword,
                result_num,
                COUNT(*) AS search_count
            FROM `user_search_logs`
            WHERE action = 'SEARCH' AND result_num = 0
            GROUP BY keyword, result_num
            ORDER BY search_count DESC;
            """
        cursor.execute(sql)
        result = cursor.fetchall()

finally:
    connection.close()

keyword_stats = {}

for item in result:
    keyword = item['keyword']
    search_count = item['search_count']
    keyword_stats[keyword] = search_count

series_search_keyword = pd.Series(keyword_stats)

fig, axes = plt.subplots(nrows=1, ncols=2)
series_search_keyword.plot(ax=axes[0], kind='bar', rot=0)
axes[0].set_title('Search No Match keyword - Bar chart')
for ax in axes.flat:
    for i, v in enumerate(ax.patches):    # 可以透過 Axes 物件中的 patches 屬性來遍歷每個子圖中的所有物件
        x = v.get_x() + v.get_width()/2   # v.get_x() 矩形的左邊緣的 x 座標值, v.get_width() 形的寬度
        y = v.get_height()
        ax.annotate(str(y), (x, y), ha='center', va='bottom')

axes[1].pie(series_search_keyword.values, labels=series_search_keyword.index, autopct='%1.1f%%')
axes[1].set_title('Search No Match keyword - Pie chart')


plt.show()