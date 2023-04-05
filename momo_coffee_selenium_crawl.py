"""
* 設定目標
  * 抓取 [MOMO購物網](https://www.momoshop.com.tw/)，並搜尋 咖啡 的清單列表的商品名稱和價格、總銷量
* 觀察網頁
  * 此頁面由 JavaScript 程式於網頁前端動態產生的內容（Ajax/XHR）, 編碼: utf-8
* 發出請求
  * 使用 requests 套件發出 GET 網路請求，取得文章列表 HTML 的內容回傳值
* 解析內容
  * 商品清單被包在 class = listArea 中的 class = goodsUrl 中
* 儲存資料成檔案
  * 將資料整理一下成為 dict 給 csv 套件寫入的資料儲存格式即可將其儲存為 momo_coffee_products.csv 檔案
"""

import time
import csv
import time
from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome('./chromedriver.exe')

driver.get('https://www.momoshop.com.tw/')
time.sleep(1)

search_input = driver.find_elements_by_css_selector('#keyword')[0]
search_btn = driver.find_elements_by_css_selector('.inputbtn')[0]
search_input.send_keys('咖啡')
search_btn.click()

# 提交查詢後等待網頁內容載入完成
time.sleep(1)

page_content = driver.page_source
soup = BeautifulSoup(page_content, 'html.parser')

# 透過 select 使用 CSS 選擇器 選取我們要選的 html 內容
elements = soup.select('.listArea .goodsUrl')
# print(elements)
row_list = []
for element in elements:
    # print(element)
    product_name = element.select('.prdInfoWrap .prdName')[0].text
    price = element.select('.price')[0].text
    amount = element.select('.prdInfoWrap .totalSales')[0].text
    print(product_name)
    print(price)
    print(amount)

    data = {}
    data['product_name'] = product_name
    data['price'] = price
    data['amount'] = amount
    row_list.append(data)


# CSV 檔案第一列標題要和 dict 的 key 相同，不然會出現錯誤
headers = ['product_name', 'price', 'amount']

with open('momo_coffee_products.csv', 'w', newline='', encoding='utf-8') as output_file:
    dict_writer = csv.DictWriter(output_file, headers)
    # 標題
    dict_writer.writeheader()
    # 值
    dict_writer.writerows(row_list)

with open('momo_coffee_products.csv', 'r', newline='', encoding='utf-8') as input_file:
    rows = csv.reader(input_file)
    
    for row in rows:
        print(row)

driver.quit()