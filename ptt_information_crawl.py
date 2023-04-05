"""
* 設定目標
  * 抓取 PTT [資訊版](https://www.ptt.cc/bbs/CVS/index.html)的文章日期、標題及作者名稱
* 觀察網頁
  * 檢視原始碼判斷是否為伺服器渲染，此網站編碼為 utf-8
* 發出請求
  * 使用 requests 套件發出 GET 網路請求，取得文章列表 HTML 的內容回傳值
* 解析內容
  * 透過 BeautifulSoup parse HTML，並使用選擇器挑出 r-ent 區塊 class 為 date, title a, author 得到文章日期、標題、作者資訊
* 儲存資料成檔案
  * 將資料轉成 dict ，再搭配 csv 模組將資料寫入 cvs_products.csv 檔案
"""

import requests
from bs4 import BeautifulSoup
import csv
import time

# 設定網路請求 headers 讓網站覺得是正常人在瀏覽
headers = {
    'user-agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
}

# 發出 requests GET
res = requests.get('https://www.ptt.cc/bbs/CVS/index.html', headers=headers)
res.encoding = 'utf-8'

# 使用 BeautifulSoup parse 內容為 BeautifulSoup 物件
soup = BeautifulSoup(res.text, 'html.parser')

# parse class name 為 r-ent 區塊 (list 列表)
items = soup.select('.r-ent')

row_list = []

# 一一抓取 class name 為 r-ent 的區塊
for item in items:
    # 取得作者
    # 為了避免抓取到刪除文章產生 list index out of range 錯誤，先判斷若文章作者若為 - 則跳過該列
    print('author', item.select('.author'))
    author = item.select('.author')[0].text
    if author == '-':
        print('continue')
        continue

    # 取得標題
    title = item.select('.title a')[0].text

    # 取得日期
    date = item.select('.date')[0].text

    # 將資料整理成一個 dict
    data = {}
    data['title'] = title
    data['author'] = author
    data['date'] = date
    row_list.append(data)
    print(data)
    time.sleep(3)

file_title = ['date', 'title', 'author']

with open('cvs_products.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, file_title)
    # 寫入標題
    dict_writer.writeheader()
    # 寫入內容(值)
    dict_writer.writerows(row_list)

with open('cvs_products.csv', 'r') as input_file:
    rows = csv.reader(input_file)

    # 以迴圈輸出每一列，每一列是一個 list
    for row in rows:
        print(row)
