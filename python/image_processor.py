from PIL import Image

im = Image.open('bulbasaur_1px.bmp') # Can be many different formats.
pix = im.load()
color = [[0 for x in range(im.size[0])] for y in range(im.size[1])]  # Get the width and hight of the image for iterating over
for x in range(0,320):
	for y in range(0,320):
		color[x][y] = pix[x,y]

for x in range(0,320):
	for y in range(0,320):
		pix[x,y] = (color[x][y][0], 0, 0)
im.save('alive_parrot_1.png')

for x in range(0,320):
	for y in range(0,320):
		pix[x,y] = (color[x][y][0],color[x][y][1], 0)
im.save('alive_parrot_2.png')


