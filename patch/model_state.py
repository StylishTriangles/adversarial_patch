import os.path as osp
import PIL.Image
import numpy as np
import tensorflow as tf
from keras import backend as K
import keras

import sys
import time

from patch.constants import *
from patch.image import *
from patch.transformations import transform_vector

image_loader = StubImageLoader(VALIDATION_DIR, BATCH_SIZE)

def get_peace_mask(shape):
  path = osp.join(DATA_DIR, "peace_sign.png")
  pic = PIL.Image.open(path)
  pic = pic.resize(shape[:2], PIL.Image.ANTIALIAS)
  if path.endswith('.png'):
    ch = 4
  else:
    ch = 3
  pic = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], ch)
  pic = pic / 127.5 - 1
  pic = pic[:,:,3]

  peace_mask = (pic + 1.0) / 2
  peace_mask = np.expand_dims(peace_mask, 2)
  peace_mask = np.broadcast_to(peace_mask, shape)
  return peace_mask


def circle_mask(shape, sharpness = 40):
  """Return a circular mask of a given shape"""
  assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

  diameter = shape[0]  
  x = np.linspace(-1, 1, diameter)
  y = np.linspace(-1, 1, diameter)
  xx, yy = np.meshgrid(x, y, sparse=True)
  z = (xx**2 + yy**2) ** sharpness

  mask = 1 - np.clip(z, -1, 1)
  mask = np.expand_dims(mask, axis=2)
  mask = np.broadcast_to(mask, shape).astype(np.float32)
  return mask

def _gen_target_ys():
  label = TARGET_LABEL
  y_one_hot = np.zeros(NUM_LABELS)
  y_one_hot[label] = 1.0
  y_one_hot = np.tile(y_one_hot, (BATCH_SIZE, 1))
  return y_one_hot

TARGET_ONEHOT = _gen_target_ys()
      
class ModelContainer():
  """Encapsulates an Imagenet model, and methods for interacting with it."""
  
  def __init__(self, model_name, verbose=True, peace_mask=None, peace_mask_overlay=0.0, custom_weights_path=None):
    """
    Args:
      peace_mask: None, "Forward", "Backward"
      custom_weights_path: If using custom models this parameter must be provided
    """
    # 
    self.model_name = model_name
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.peace_mask = peace_mask
    self.patch_shape = PATCH_SHAPE
    self._peace_mask_overlay = peace_mask_overlay
    self.custom_weights_path = custom_weights_path
    self.load_model(verbose=verbose)

    
  def patch(self, new_patch=None):
    """Retrieve or set the adversarial patch.
    
    new_patch: The new patch to set, or None to get current patch.
    
    Returns: Itself if it set a new patch, or the current patch."""
    if new_patch is None:
      return self._run(self._clipped_patch)
      
    self._run(self._assign_patch, {self._patch_placeholder: new_patch})
    return self
  
  
  def reset_patch(self):
    """Reset the adversarial patch to all zeros."""
    self.patch(np.zeros(self.patch_shape))
    
  def train_step(self, images=None, target_ys=None, learning_rate=5.0, scale=(0.1, 1.0), dropout=None, patch_disguise=None, disguise_alpha=None):
    """Train the model for one step.
    
    Args:
      images: A batch of images to train on, it loads one if not present.
      target_ys: Onehot target vector, defaults to TARGET_ONEHOT
      learning_rate: Learning rate for this train step.
      scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

    Returns: Loss on the target ys."""
    if images is None:
      images = image_loader.get_images()
    if target_ys is None:
      target_ys = TARGET_ONEHOT
    
    feed_dict =  {self._image_input: images, 
                        self._target_ys: target_ys,
                        self._learning_rate: learning_rate}
    
    if patch_disguise is not None:
      if disguise_alpha is None:
        raise ValueError("You need disguise_alpha")
      feed_dict[self.patch_disguise] = patch_disguise
      feed_dict[self.disguise_alpha] = disguise_alpha
    

    loss, _ = self._run([self._loss, self._train_op], feed_dict, scale=scale, dropout=dropout) 
    return loss
  
  def inference_batch(self, images=None, target_ys=None, scale=None):
    """Report loss and label probabilities, and patched images for a batch.
    
    Args:
      images: A batch of images to train on, it loads if not present.
      target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present."""
    if images is None:
      images = image_loader.get_images()
    if target_ys is None:
      target_ys = TARGET_ONEHOT
      
    feed_dict = {self._image_input: images, self._target_ys: target_ys}

    loss_per_example, ps, ims = self._run([self._loss_per_example, self._probabilities, self._patched_input],
                            feed_dict, scale=scale)
    return loss_per_example, ps, ims
  
  def load_model(self, verbose=True):
   
    model = NAME_TO_MODEL[self.model_name]
    if self.model_name in ['xception', 'inceptionv3', 'mobilenet']:
      keras_mode = False
    else:
      keras_mode = True
    patch = None

    self._make_model_and_ops(model, keras_mode, patch, verbose)
            
  def _run(self, target, feed_dict=None, scale=None, dropout=None):
    K.set_session(self.sess)
    if feed_dict is None:
      feed_dict = {}
    feed_dict[self.learning_phase] = False
    
    if scale is not None:
      if isinstance(scale, (tuple, list)):
        scale_min, scale_max = scale
      else:
        scale_min, scale_max = (scale, scale)
      feed_dict[self.scale_min] = scale_min
      feed_dict[self.scale_max] = scale_max
         
    if dropout is not None:
      feed_dict[self.dropout] = dropout
    
    # print(feed_dict)
    return self.sess.run(target, feed_dict=feed_dict)
  
  
  def _make_model_and_ops(self, M, keras_mode, patch_val, verbose):
    start = time.time()
    K.set_session(self.sess)
    with self.sess.graph.as_default():
      self.learning_phase = K.learning_phase()

      image_shape = INPUT_SHAPE
      self._image_input = keras.layers.Input(shape=image_shape)
      
      self.scale_min = tf.placeholder_with_default(SCALE_MIN, [])
      self.scale_max = tf.placeholder_with_default(SCALE_MAX, [])
      self._scales = tf.random_uniform([BATCH_SIZE], minval=self.scale_min, maxval=self.scale_max)

      image_input = self._image_input
      self.patch_disguise = tf.placeholder_with_default(tf.zeros(self.patch_shape), shape=self.patch_shape)
      self.disguise_alpha = tf.placeholder_with_default(0.0, [])
      patch = tf.get_variable("patch", self.patch_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
      self._patch_placeholder = tf.placeholder(dtype=tf.float32, shape=self.patch_shape)
      self._assign_patch = tf.assign(patch, self._patch_placeholder)

      modified_patch = patch

      def clip_to_valid_image(x):    
        return tf.clip_by_value(x, clip_value_min=-1.,clip_value_max=1.)

      if self.peace_mask == 'forward':
        mask = get_peace_mask(self.patch_shape)
        modified_patch = patch * (1 - mask) - np.ones(self.patch_shape) * mask + (1+patch) * mask * self._peace_mask_overlay

      self._clipped_patch = clip_to_valid_image(modified_patch)
      
      if keras_mode:
        image_input = tf.image.resize_images(image_input, INPUT_SIZE)
        image_shape = INPUT_SHAPE
        modified_patch = tf.image.resize_images(patch, PATCH_SIZE)
      
      self.dropout = tf.placeholder_with_default(1.0, [])
      patch_with_dropout = tf.nn.dropout(modified_patch, keep_prob=self.dropout)
      patched_input = clip_to_valid_image(self._random_overlay(image_input, patch_with_dropout, image_shape))


      def to_keras(x):
        x = (x + 1) * 127.5
        R,G,B = tf.split(x, 3, 3)
        R -= 123.68
        G -= 116.779
        B -= 103.939
        x = tf.concat([B,G,R], 3)

        return x

      # Since this is a return point, we do it before the Keras color shifts
      # (but after the resize, so we can see what is really going on)
      self._patched_input = patched_input

      if keras_mode:
        patched_input = to_keras(patched_input)


      # Labels for our attack (e.g. always a toaster)
      self._target_ys = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))

      if self.model_name in MODELS:
        model = M(input_tensor=patched_input, classes=NUM_LABELS, input_shape=INPUT_SHAPE)
        if self.custom_weights_path is None:
          raise Exception(
            "When using custom models, custom_weights_path (a path to an .h5 file containing weights) "
            "must be provided during initialization of ModelContainer"
          )
        model.load_weights(self.custom_weights_path)
      else:
        model = M(input_tensor=patched_input, weights='imagenet')

      # Pre-softmax logits of our pretrained model
      logits = model.outputs[0].op.inputs[0]

      self._loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
          labels=self._target_ys, 
          logits=logits
      )
      self._target_loss = tf.reduce_mean(self._loss_per_example)
      
      self._patch_loss = tf.nn.l2_loss(patch - self.patch_disguise) * self.disguise_alpha
      
      
      
      self._loss = self._target_loss + self._patch_loss

      # Train our attack by only training on the patch variable
      self._learning_rate = tf.placeholder(tf.float32)
      self._train_op = tf.train.GradientDescentOptimizer(self._learning_rate)\
                               .minimize(self._loss, var_list=[patch])

      self._probabilities = model.outputs[0]

      if patch_val is not None:
        self.patch(patch_val)
      else:
        self.reset_patch()


      elapsed = time.time() - start
      if verbose:
        print("Finished loading {}, took {:.0f}s".format(self.model_name, elapsed))       


  def _pad_and_tile_patch(self, patch, image_shape):
    # Calculate the exact padding
    # Image shape req'd because it is sometimes 299 sometimes 224
    
    # padding is the amount of space available on either side of the centered patch
    # WARNING: This has been integer-rounded and could be off by one. 
    #          See _pad_and_tile_patch for usage
    return tf.stack([patch] * BATCH_SIZE)

  def _random_overlay(self, imgs, patch, image_shape):
    """Augment images with random rotation, transformation.

    Image: BATCHx299x299x3
    Patch: 50x50x3

    """
    # Add padding
    
    image_mask = circle_mask(image_shape)

    if self.peace_mask == 'backward':
      peace_mask = get_peace_mask(image_shape)
      image_mask = (image_mask * peace_mask).astype(np.float32)
    image_mask = tf.stack([image_mask] * BATCH_SIZE)
    padded_patch = tf.stack([patch] * BATCH_SIZE)

    transform_vecs = []    
    
    def _random_transformation(scale_min, scale_max, width):
      im_scale = np.random.uniform(low=scale_min, high=scale_max)

      padding_after_scaling = (1-im_scale) * width
      x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
      y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)


      rot = np.random.uniform(-MAX_ROTATION, MAX_ROTATION)

      return transform_vector(width, 
                                       x_shift=x_delta,
                                       y_shift=y_delta,
                                       im_scale=im_scale, 
                                       rot_in_degrees=rot)    

    for i in range(BATCH_SIZE):
      # Shift and scale the patch for each image in the batch
      random_xform_vector = tf.py_func(_random_transformation, [self.scale_min, self.scale_max, image_shape[0]], tf.float32)
      random_xform_vector.set_shape([8])

      transform_vecs.append(random_xform_vector)

    image_mask = tf.contrib.image.transform(image_mask, transform_vecs, "BILINEAR")
    padded_patch = tf.contrib.image.transform(padded_patch, transform_vecs, "BILINEAR")

    inverted_mask = (1 - image_mask)
    return imgs * inverted_mask + padded_patch * image_mask
  

class MetaModel():
  def __init__(self, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
    self.nc = {m: ModelContainer(m, verbose=verbose, peace_mask=peace_mask, peace_mask_overlay=peace_mask_overlay) for m in MODEL_NAMES}
    self._patch = np.zeros(PATCH_SHAPE)
    self.patch_shape = PATCH_SHAPE
        
  def patch(self, new_patch=None):
    """Retrieve or set the adversarial patch.
    
    new_patch: The new patch to set, or None to get current patch.
    
    Returns: Itself if it set a new patch, or the current patch."""
    if new_patch is None:
      return self._patch
    
    self._patch = new_patch
    return self
  
  def reset_patch(self):
    """Reset the adversarial patch to all zeros."""
    self.patch(np.zeros(self.patch_shape))
    
  def train_step(self, model=None, steps=1, images=None, target_ys=None, learning_rate=5.0, scale=None, **kwargs):
    """Train the model for `steps` steps.
    
    Args:
      images: A batch of images to train on, it loads one if not present.
      target_ys: Onehot target vector, defaults to TARGET_ONEHOT
      learning_rate: Learning rate for this train step.
      scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
      
    Returns: Loss on the target ys."""
    
    
    if model is not None:
      to_train = [self.nc[model]]
    else:
      to_train = self.nc.values()
      
    losses = []
    for mc in to_train:
      mc.patch(self.patch())
      for i in range(steps): 
        loss = mc.train_step(images, target_ys, learning_rate, scale=scale, **kwargs)
        losses.append(loss)
      self.patch(mc.patch())
    return np.mean(losses)
  
  def inference_batch(self, model, images=None, target_ys=None, scale=None):
    """Report loss and label probabilities, and patched images for a batch.
    
    Args:
      images: A batch of images to train on, it loads if not present.
      target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present.
      scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
    """

    mc = self.nc[model]
    mc.patch(self.patch())
    return mc.inference_batch(images, target_ys, scale=scale)