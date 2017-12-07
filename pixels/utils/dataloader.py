import threading
import random
import time

import tensorflow as tf
import numpy as np


def dtype(item):
  if isinstance(item, np.ndarray):
    return item.dtype
  else:
    return np.dtype(type(item))

def shape(item):
  if isinstance(item, np.ndarray):
    return item.shape
  else:
    return np.shape(item)

def _determine_metadata(dataset):
  sample = dataset[0]
  if isinstance(sample, (tuple, list)):
    dtypes = [ dtype(it) for it in sample ]
    shapes = [ shape(it) for it in sample ]
    return type(sample), dtypes, shapes, None
  elif isinstance(sample, dict):
    dtypes = [ dtype(sample[it]) for it in sample.keys() ]
    shapes = [ shape(sample[it]) for it in sample.keys() ]
    return dict, dtypes, shapes, sample.keys()
  else:
    return type(sample), [dtype(sample)], [shape(sample)], None

class DataLoader(object):

  def __init__(self, dataset, batch_size, cycle=True,
         shuffle=True, num_workers=4):
    self.batch_size = batch_size
    self.cycle = cycle
    self.shuffle = shuffle
    self.num_workers = num_workers
    #sself.batch = self._wrap_batch(dataset)
    stype, dtypes, shapes, keys = _determine_metadata(dataset)
    self.tensors = dict(zip(
        keys, 
        [ tf.placeholder(dtypes[i], shape=shapes[i]) 
          for i in range(len(dtypes)) ])
      )

    self.queue = tf.FIFOQueue(batch_size * 3, dtypes, shapes=shapes)
    self.enqueue_op = self.queue.enqueue(self.tensors.values())

    self.stype = stype
    self.keys = keys
    self.dataset = dataset
    self.shuffle = shuffle
    self.num_samples = len(dataset)
    self.enqueue_lock = threading.RLock()
    self.fetch_sample_lock = threading.RLock()
    self.curr_sample = 0
    self.curr_enqueue_idx = 0
    self.batch = self._make_batch()

    if self.shuffle:
      self.sample_idx = random.shuffle(range(self.num_samples))
    else:
      self.sample_idx = range(self.num_samples)

  def _enqueue_sample(self, sess, idx, sample):
    feed_dict = dict()
    for k in sample:
      feed_dict[self.tensors[k]] = sample[k]

    if self.shuffle:
      with self.enqueue_lock:
        sess.run(self.enqueue_op, feed_dict=feed_dict)
    else:
      while True:
        with self.enqueue_lock:
          if self.curr_enqueue_idx == idx: 
            self.queue.enqueue(sample)
            self.curr_enqueue_idx = (idx + 1) % self.num_samples
            break
        time.sleep(1)

  def _fetch_sample_index(self):
    with self.fetch_sample_lock:
      if self.curr_sample >= self.num_samples:
        if self.cycle:
          self.curr_sample = 0
        else:
          return None
      key = self.sample_idx[self.curr_sample]
      self.curr_sample += 1
    return key

  def _make_batch(self):
    batch = self.queue.dequeue_many(self.batch_size)
    stype = self.stype
    if stype == list or stype == tuple:
      return stype(batch)
    elif stype == dict:
      return dict(zip(self.keys, batch))
    else:
      return batch[0]

  def start_queue_runners(self, sess, coord):
    # enqueuing batches procedure
    def enqueue_batches():
      while not coord.should_stop():
        key = self._fetch_sample_index()
        if key is not None:
          sample = self.dataset[key]
          self._enqueue_sample(sess, key, sample)
        else:
          time.sleep(1)

    # creating and starting parallel threads to fill the queue
    for i in range(self.num_workers):
      t = threading.Thread(target=enqueue_batches)
      t.setDaemon(True)
      t.start()
  
  def __getitem__(self, key):
    if key in self.batch:
      return self.batch[key]
    return None