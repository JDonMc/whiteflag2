from english_words import english_words_set
import requests
import bs4

def get_book_page_urls(index_url):
    response = requests.get(index_url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    return [a.attrs.get('href') for a in soup.select('li.booklink a[href]')]

def get_book_files(index_url):
	response = requests.get(index_url)
	soup = bs4.BeautifulSoup(response.text, 'html.parser')
	return [a.attrs.get('href') for a in soup.select('tr.even td.unpadded a.link[href]')]

def get_html_source_folder(index_url):
	response = requests.get(index_url)
	soup = bs4.BeautifulSoup(response.text, 'html.parser')
	return [a.attrs.get('href') for a in soup.select('tr td a[href]')]

def get_html(index_url):
	response = requests.get(index_url)
	soup = bs4.BeautifulSoup(response.text, 'html.parser')
	return [a.attrs.get('href') for a in soup.select('tr td a[href]')]


for word in english_words_set:
	base = "https://www.gutenberg.org/ebooks/search/?submit_search=Search&query="
	base += word
	print(word)
	r = requests.post('https://www.babylonpolice.com/B/words/',data={'the_word_itself':word})
	print(r.status_code)
	print(r.text)
	for a in set(get_book_page_urls(base)):
		print(a)
		book_page = "https://www.gutenberg.org"
		if len(a)<40:
			print(book_page+a)
			for b in set(get_book_files(book_page+a)):
				print(b)
				url = book_page +b
				
				if url.endswith('.txt') or url.endswith('.txt.utf-8'):
					r = requests.get(url)
					title = bs4.BeautifulSoup(r.content, 'html.parser').prettify()[0:144]
					body = bs4.BeautifulSoup(r.content, 'html.parser').prettify()[144:]
					print(title)
					if body:
						chapter_counter = 0
						body_len = len(body)
						for chapter in range(0, body_len, 144000):
							chapter_counter += 1
							r = requests.post('https://www.babylonpolice.com/B/posts/',data={'title':title[0:100] + ' Chapter '+ str(chapter_counter), "body":body[(chapter_counter-1)*144000:chapter_counter*144000]})
							print(r.status_code)
							print(r.text)