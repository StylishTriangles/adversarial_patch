import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import os
import random

from patch.constants import PATCH_SIZE

def _convert(im: PIL.Image):
  return ((im + 1) * 127.5).astype(np.uint8)

def show(im: PIL.Image):
  plt.axis('off')
  plt.imshow(_convert(im), interpolation="nearest")
  plt.show()
  
def load_image(image_path: str):
  im = PIL.Image.open(image_path)
  im = im.resize(PATCH_SIZE, PIL.Image.ANTIALIAS)
  if image_path.endswith('.png'):
    ch = 4
  else:
    ch = 3
  try:
    im = np.array(im.getdata()).reshape(im.size[0], im.size[1], ch)[:,:,:3]
  except ValueError as e:
    print("An error ocurred when processing file", image_path)
    raise e
  return im / 127.5 - 1


class StubImageLoader():
  """An image loader that uses just a few ImageNet-like images. 
  In fact, all images are supplied by the user.
  """
  def __init__(self, images_dir, batch_size):
    self.image_paths = []
    self.batch_size = batch_size
   
    #only keep the image paths and load them when requested
    for dirpath, _, filenames in os.walk(images_dir):
      for image_path in filenames:
        self.image_paths.append(os.path.join(dirpath, image_path))

  def get_images(self):
    # fetch a random sample of images
    chosen = random.sample(self.image_paths, self.batch_size)

    return [load_image(img_path) for img_path in chosen]
