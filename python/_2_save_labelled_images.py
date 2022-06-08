from english_words import english_words_set
import requests
import bs4
import urllib.request

def get_search_page_images(index_url):
    response = requests.get(index_url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    return [a.attrs.get('src') for a in soup.select('img.h-32')]



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
	    