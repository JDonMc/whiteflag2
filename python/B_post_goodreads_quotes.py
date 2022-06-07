from os import listdir
from os.path import isfile, join

import requests
import json

mypath = "/Users/adenhandasyde/GitHub/GoodReads/"
onlyfiles = [join(f, g) for f in listdir(mypath) for g in listdir(mypath+f) if isfile(join(join(mypath, f), g))]
print(onlyfiles)

for file in onlyfiles:
	path = mypath + file
	writefile = open(path, 'r')
	jsonfile = json.load(writefile)
	print(jsonfile)
	print(jsonfile['author'])
	r = requests.post('https://www.babylonpolice.com/B/examples/',data={'the_example_itself':jsonfile['text'], "author":jsonfile['author']})
	print(r.status_code)
	print(r.text)

    