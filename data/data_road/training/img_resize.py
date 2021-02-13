from PIL import Image
import os, sys

path = "train/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((1024,540), Image.ANTIALIAS)
            imResize.save(f + '.png', 'png', quality=90)
            print('Generating images',item)

resize()