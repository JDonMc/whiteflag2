rss_feed_urls = ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss", "https://news.bitcoin.com/feed/", "https://cryptopotato.com/feed/", "https://zycrypto.com/category/news/feed/", "https://nulltx.com/feed/", "https://coinquora.com/news/feed/", "https://ambcrypto.com/feed/", "https://cryptoslate.com/feed/", "https://crypto.news/feed/"]

politics_rss = ["http://www.politicususa.com/feed", "https://www.thegatewaypundit.com/feed/", "https://thepoliticalinsider.com/feed/", "https://www.politico.com/", "https://reason.com/feed/", "http://www.realclearpolitics.com/index.xml", "http://www.dailykos.com/blogs/main.rss", "https://politicalwire.com/feed/", "https://front.moveon.org/feed/", "http://www.vox.com/rss/policy-and-politics/index.xml",
                "https://medium.com/feed/voterly", "https://feeds.megaphone.fm/slatespoliticalgabfest", "https://www.theatlantic.com/feed/channel/politics/", "https://blog.feedspot.com/political_rss_feeds/", "https://babyboomerresistance.home.blog/feed/", "https://fivethirtyeight.com/politics/feed/", "https://redstate.com/feed/", "https://boingboing.net/tag/politics/feed", ]

from feedparser import parse
#feed = parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
#print(feed)
import requests
for url in politics_rss:
    feed = parse(url)
    print(feed)
    for entry in feed['entries']:
        title = entry['title']
        body = entry['summary']
        r = requests.post('https://www.babylonpolice.com/B/posts/',data={'title':title, "body":body})
        print(r.status_code)
        print(r.text)

