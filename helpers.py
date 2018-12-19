import numpy as np
import csv

def read_csv(path):
  width = 34
  height = 26
  dims = 1

  with open(path,'r') as f:
    #read the scv file with the dictionary format
    reader = csv.DictReader(f)
    rows = list(reader)

  #imgs is a numpy array with all the images
  #tgs is a numpy array with the tags of the images
  imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
  tgs = np.empty((len(list(rows)),1))

  for row,i in zip(rows,range(len(rows))):
    #convert the list back to the image format
    img = row['image']
    img = img.strip('[').strip(']').split(', ')
    im = np.array(img,dtype=np.uint8)
    im = im.reshape((height, width))
    im = np.expand_dims(im, axis=2)
    imgs[i] = im

    #the tag for open is 1 and for close is 0
    tag = row['state']
    if tag == 'open':
      tgs[i] = 1
    else:
      tgs[i] = 0

  #shuffle the dataset
  index = np.random.permutation(imgs.shape[0])
  imgs = imgs[index]
  tgs = tgs[index]

  #return images and their respective tags
  return imgs, tgs