from __future__ import division
import argparse

from pixels.dataset.kitti import KittiRaw
from pixels.utils.dataloader import DataLoader
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description="Monodepth Pixels implementation.")
parser.add_argument('--mode', type=str, default='train',
    help='The mode of the process, train or test')
parser.add_argument('--data_name', type=str, default="kitti_raw",
          help='The dataset name used for training')
parser.add_argument('data_dir', type=str,
          help='Path to the training data directory')
parser.add_argument('--include_list_file', type=str, default='',
          help='The list file for training frame in KITTI')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Maximum number of training epochs')
parser.add_argument('--num_gpus', type=int, default=1,
          help='The number of gpus to run')
parser.add_argument('--num_workers', type=int, default=8,
          help='The number workers for loading data')
parser.add_argument("--checkpoint_dir", type=str, default="ckpt/", 
          help="Directory name to save the checkpoints")
parser.add_argument("--learning_rate", type=float, default=1e-4, 
          help="Initial learning rate")
parser.add_argument("--encoder", type=str, default="vgg", 
          help="Type of encoder, vgg or resnet50")
parser.add_argument("--lr_loss_weight", type=float, default=1.0,
          help="Left-right consistency weight")
parser.add_argument('--alpha_image_loss', type=float, default=0.85,
          help='Weight between SSIM and L1 in the image loss')
parser.add_argument("--batch_size", type=int, default=8, 
          help="The size of of a sample batch")
parser.add_argument("--image_height", type=int, default=256, 
          help="Image height")
parser.add_argument("--image_width", type=int, default=512, 
          help="Image width")
parser.add_argument('--disp_gradient_loss_weight', type=float,
          default=0.1, help='Disparity smoothness weigth')
parser.add_argument("--summary_freq", type=int, default=100, 
          help="Logging every log_freq iterations")
parser.add_argument("--save_latest_freq", type=int, default=5000,
          help="""
            Save (and overwrite) the latest model 
            every save_latest_freq iterations
          """)
parser.add_argument("--continue_train", type=bool, default=False,
          help="Continue training from previous checkpoint")
parser.add_argument('--wrap_mode', type=str, default='border',
          help='Bilinear sampler wrap mode, edge or border')
parser.add_argument('--retrain', action='store_true',
          help="""
          If used with checkpoint_path, will restart 
          training from step zero
          """)
parser.add_argument('--log_directory', type=str, default='',
          help="Directory to save checkpoints and summaries")
parser.add_argument('--full_summary', action='store_true',
          help="""
          If set, will keep more data for each summary.
          Warning: the file can become very large
          """)
parser.add_argument("--use_deconv", action="store_true",
          help="If set, will use transposed convolutions")
args = parser.parse_args()

def main(_):
  with tf.Graph().as_default(), tf.device("/cpu:0"):
    global_step = tf.Variable(0, trainable=False)

    dataset = KittiRaw(args.data_dir, model_type="stereo", 
        include_list_file=args.include_list_file)

    data_loader = DataLoader(dataset, args.batch_size)
    num_training_samples = len(dataset)

    steps_per_epoch = np.ceil(
      num_training_samples / args.batch_size
    ).astype(np.int32)
    num_total_steps = args.num_epochs * steps_per_epoch
    start_learning_rate = args.learning_rate

    boundaries = [
      np.int32((3/5) * num_total_steps), 
      np.int32((4/5) * num_total_steps)
    ]
    values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    opt_step = tf.train.AdamOptimizer(learning_rate)
    print("total number of samples: {}".format(num_training_samples))
    print("total number of steps: {}".format(num_total_steps))

    left, right = data_loader.batch["left"], data_loader.batch["right"]

    left_splits = tf.split(left, args.num_gpus, 0)
    right_splits = tf.split(right, args.num_gpus, 0)

    tower_grads  = []
    tower_losses = []
    reuse_variables = None
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(args.num_gpus):
        with tf.device('/gpu:%d' % i):

          model = MonodepthModel(params, left_splits[i], right_splits[i], 
            reuse_variables, i)

          loss = model.total_loss
          tower_losses.append(loss)

          reuse_variables = True

          grads = opt_step.compute_gradients(loss)

          tower_grads.append(grads)

    grads = average_gradients(tower_grads)

    apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

    total_loss = tf.reduce_mean(tower_losses)

    tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
    tf.summary.scalar('total_loss', total_loss, ['model_0'])
    summary_op = tf.summary.merge_all('model_0')

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    summary_writer = tf.summary.FileWriter(
      args.log_directory + '/monodepth', sess.graph)
    train_saver = tf.train.Saver()

    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.trainable_variables():
      total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(total_num_parameters))

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # LOAD CHECKPOINT IF SET
    if args.checkpoint_path != '':
      train_saver.restore(sess, args.checkpoint_path.split(".")[0])

    if args.retrain:
      sess.run(global_step.assign(0))

    # GO!
    start_step = global_step.eval(session=sess)
    start_time = time.time()
    for step in range(start_step, num_total_steps):
      before_op_time = time.time()
      _, loss_value = sess.run([apply_gradient_op, total_loss])
      duration = time.time() - before_op_time
      if step and step % 100 == 0:
        examples_per_sec = params.batch_size / duration
        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (num_total_steps / step - 1.0) * time_sofar
        print_string = """
            batch {:>6} | examples/s: {:4.2f} | 
            loss: {:.5f} | time elapsed: {:.2f}h | 
            time left: {:.2f}h
        """
        print(print_string.format(
          step, examples_per_sec, loss_value, time_sofar, training_time_left))
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, global_step=step)
      if step and step % 10000 == 0:
        train_saver.save(sess, args.log_directory + '/monodepth/model',
          global_step=step)

    train_saver.save(sess, args.log_directory + '/monodepth/model', 
      global_step=num_total_steps)

if __name__ == "__main__":
  tf.app.run()
