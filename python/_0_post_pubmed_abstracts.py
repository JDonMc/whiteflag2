from english_words import english_words_set
import requests
import bs4

def get_video_page_urls(index_url):
    response = requests.get(index_url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    return [a.attrs.get('href') for a in soup.select('div.docsum-content a[href]')]


for word in english_words_set:
	base = "https://pubmed.ncbi.nlm.nih.gov/pubmed/?term="
	base += word
	print(word)
	r = requests.post('https://www.babylonpolice.com/B/words/',data={'the_word_itself':word})
	print(r.status_code)
	print(r.text)
	for a in set(get_video_page_urls(base)):
	    print(a)
	    if len(a)<40:
	        url="https://pubmed.ncbi.nlm.nih.gov" + a
	        print(url)
	        r = requests.get(url)
	        title = bs4.BeautifulSoup(r.content, 'html.parser').select('h1.heading-title')[0].prettify()
	        body = bs4.BeautifulSoup(r.content, 'html.parser').select('div.abstract-content')
	        print(title)
	        if body:
	            r = requests.post('https://www.babylonpolice.com/B/posts/',data={'title':title[0:200], "body":body[0].prettify()[0:144000]})
	            print(r.status_code)
	            print(r.text)