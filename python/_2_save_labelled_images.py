from english_words import english_words_set
import requests
import bs4
import urllib.request
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time

driver = webdriver.Chrome(ChromeDriverManager().install())

def get_search_page_images(index_url):
	headers = { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36"}

	response = driver.get(index_url)
	time.sleep(0.5)
	soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
	print(soup.prettify())
	return [a['src'] for a in soup.find_all('img.mimg')]





for word in english_words_set:
	base = "https://presearch.com/images?q="
	base += word
	print(word)
	r = requests.post('https://www.babylonpolice.com/B/words/',data={'the_word_itself':word})
	print(r.status_code)
	print(r.text)
	count = 0
	for a in set(get_search_page_images(base)):
		print(a)
		urllib.request.urlretrieve(a, word + "-" + str(count) + ".jpg")
		count += 1
	    