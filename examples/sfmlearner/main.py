import argparse
import time
import math

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib import training
from pixels.utils.dataloader import DataLoader
from pixels.datasets.kitti import *
from pixels.utils.multi_gpus import average_gradients
from pixels.utils.misc import *

from model import SFMLearner

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
    help='The mode of the process, train, test_depth, test_odom')
parser.add_argument('--dataset', type=str, default="kitti_raw",
    help='The dataset name used for training')
parser.add_argument('data_dir', type=str,
    help='Path to the training data directory')
parser.add_argument('--kitti_raw_static_list', type=str, default=None,
          help='The list file for static frame in KITTI, which should be masked')
parser.add_argument('--max_steps', type=int, default=200000,
                    help='Maximum number of training iterations')
parser.add_argument('--num_gpus', type=int, default=1,
          help='The number of gpus to run')
parser.add_argument('--num_workers', type=int, default=8,
          help='The number workers for loading data')
parser.add_argument("--checkpoint_dir", type=str, default="ckpt/", 
          help="Directory name to save the checkpoints")
parser.add_argument("--learning_rate", type=float, default=0.0002, 
          help="Learning rate of for adam")
parser.add_argument("--beta1", type=float, default=0.9, 
          help="Momentum term of adam")
parser.add_argument("--smooth_weight", type=float, default=0.5,
          help="Weight for smoothness")
parser.add_argument("--batch_size", type=int, default=4, 
          help="The size of of a sample batch")
parser.add_argument("--image_height", type=int, default=128, 
          help="Image height")
parser.add_argument("--image_width", type=int, default=384, 
          help="Image width")
parser.add_argument("--frames_length", type=int, default=3, 
          help="Sequence length for each example")
parser.add_argument("--summary_freq", type=int, default=100, 
          help="Logging every log_freq iterations")
parser.add_argument("--save_latest_freq", type=int, default=5000,
            help="Save (and overwrite) the latest model every save_latest_freq iterations")
parser.add_argument("--continue_train", type=bool, default=False, 
          help="Continue training from previous checkpoint")
params = parser.parse_args()

TOWER_NAME = 'tower'


def test_depth(params):
  """Test depth function."""
  if params.dataset == 'kitti_raw':
    dataset = KittiRaw(params.data_dir, params.image_height, params.image_width,
        model_type='mono', frames_length=params.frames_length, 
        exclude_list_file=params.kitti_raw_static_list, 
        data_list=['image', 'intrinsic'])
  elif params.dataset == 'kitti_eigen':
    dataset = KittiEigen(params.data_dir, params.image_height, params.image_width,
        data_list=['mono', 'depth_gt'])

  loader = DataLoader(dataset, 1, num_workers=params.num_workers, cycle=False, shuffle=False)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  
  images = loader['mono']
  model = SFMLearner('test_depth', images, None, num_scales=1)
  
  saver = tf.train.Saver(tf.model_variables() + [global_step])
  sv = tf.train.Supervisor(logdir=params.checkpoint_dir, save_summaries_secs=0, saver=None)
  config = tf.ConfigProto(allow_soft_placement=True)

  with sv.managed_session(config=config) as sess:
    # INIT
    #initialize_uninitialized(sess)
    coordinator = tf.train.Coordinator()
    loader.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = tf.train.latest_checkpoint(params.checkpoint_dir)
    saver.restore(sess, restore_path)

    num_test = len(dataset)

    print('now testing {} files'.format(num_test))

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    for step in range(num_test):
      depth = sess.run(model.pred_depths[0][0])
      depth_gt = dataset[step]["depth_gt"]
      depth = cv2.resize(depth, depth_gt.shape, 
                       interpolation=cv2.INTER_LINEAR)
      depth = np.squeeze(depth, axis=2)
      abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
          compute_errors(depth, depth_gt)

  print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
    'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
  print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f},"
    " {:10.4f}".format(
      abs_rel.mean(), 
      sq_rel.mean(), 
      rms.mean(), 
      log_rms.mean(), 
      d1_all.mean(), 
      a1.mean(), 
      a2.mean(), 
      a3.mean()
    ))

  print('done.')


def train(params):
  if params.dataset == 'kitti_raw':
    dataset = KittiRaw(args.data_dir, params.image_height, params.image_width,
        model_type='mono', frames_length=params.frames_length, 
        exclude_list_file=params.kitti_raw_static_list, 
        data_list=['image', 'intrinsic'])
  else:
    dataset = KittiEigen(params.data_dir, params.image_height, params.image_width,
        data_list=['mono', 'depth'])

  data_loader = DataLoader(dataset, params.batch_size * params.num_gpus, 
        num_workers=params.num_workers)

  steps_per_epoch = len(train_dataset) / (params.num_gpus * params.batch_size)

  global_step = tf.Variable(0, name='global_step', trainable=False)

  optim = tf.train.AdamOptimizer(params.learning_rate, params.beta1)

  inputs = data_loader['image']
  image_splits  = tf.split(inputs['image'],  params.num_gpus, 0)
  intrinsic_splits = tf.split(inputs['intrinsic'],  params.num_gpus, 0)

  tower_grads = []
  tower_losses = []
  reuse_variables = None
  with tf.variable_scope(tf.get_variable_scope()):
    for i in xrange(params.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (TOWER_NAME, i)):

          model = SFMLearner(image_splits[i], 
              tf.cast(intrinsic_splits[i], 'float32'), reuse=reuse_variables)

          reuse_variables = True
          loss = model.total_loss
          tower_losses.append(loss)

          grad = optim.compute_gradients(loss, )
          tower_grads.append(grad)

  grads = average_gradients(tower_grads)
  apply_gradient_op = optim.apply_gradients(grads, global_step=global_step)
  total_loss = tf.reduce_mean(tower_losses)

  training.add_gradients_summaries(grads)

  saver = tf.train.Saver(tf.model_variables() + [global_step])
  sv = tf.train.Supervisor(logdir=params.checkpoint_dir, save_summaries_secs=0, saver=None)
  config = tf.ConfigProto(allow_soft_placement=True)

  with sv.managed_session(config=config) as sess:
    # INIT
    #initialize_uninitialized(sess)
    coordinator = tf.train.Coordinator()
    train_loader.start_queue_runners(sess=sess, coord=coordinator)

    if params.continue_train:
      if params.init_checkpoint_file is None:
        checkpoint = tf.train.latest_checkpoint(params.checkpoint_dir)
      else:
        checkpoint = params.init_checkpoint_file
        print("Resume training from previous checkpoint: %s" % checkpoint)
        saver.restore(sess, checkpoint)

    start_time = time.time()
    for step in range(1, params.max_steps):
      fetches = {
        "train": apply_gradient_op,
        "global_step": global_step,
      }
      
      if step % params.summary_freq == 0:
        fetches["loss"] = total_loss
        fetches["summary"] = sv.summary_op

      results = sess.run(fetches)
      gs = results["global_step"]

      if step % params.summary_freq == 0:
        sv.summary_writer.add_summary(results["summary"], gs)
        train_epoch = (gs + steps_per_epoch - 1) / steps_per_epoch
        train_step = gs - (train_epoch - 1) * steps_per_epoch
        print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
            % (train_epoch, train_step, steps_per_epoch, \
              (time.time() - start_time)/params.summary_freq, results["loss"]))
        start_time = time.time()

      if step % params.save_latest_freq == 0:
        saver.save(sess, params.checkpoint_dir, global_step, 'latest')

def main(_):
  if params.mode == 'train':
    train(params)
  elif params.mode == 'test_depth':
    test_depth(params)

if __name__ == "__main__":
  tf.app.run()