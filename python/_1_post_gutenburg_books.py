from english_words import english_words_set
import requests
import bs4

def get_video_page_urls(index_url):
    response = requests.get(index_url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    return [a.attrs.get('href') for a in soup.select('li.booklink a[href]')]



for word in english_words_set:
	base = "https://www.gutenberg.org/ebooks/search/?submit_search=Search&query="
	base += word
	print(word)
	r = requests.post('https://www.babylonpolice.com/B/words/',data={'the_word_itself':word})
	print(r.status_code)
	print(r.text)
	for a in set(get_video_page_urls(base)):
	    print(a)
	    if len(a)<40:
	        url="https://www.gutenberg.org/files/" + a[8:] + '/' + a[8:] + '-h/' + a[8:] + '-h.html'

	        
	        print(url)
	        r = requests.get(url)
	        title = bs4.BeautifulSoup(r.content, 'html.parser').select('div')[0].prettify()
	        body = bs4.BeautifulSoup(r.content, 'html.parser').select('div.chapter')
	        print(title)
	        if body:
	        	chapter_counter = 0
	        	for chapter in body:
	        		chapter_counter += 1
	        		r = requests.post('https://www.babylonpolice.com/B/posts/',data={'title':title[0:150] + ' Chapter '+ str(chapter_counter), "body":chapter[chapter_counter].prettify()[0:144000]})
		            print(r.status_code)
		            print(r.text)