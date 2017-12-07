from __future__ import division

import os
import glob
import collections
import random
import os
import cv2
import pickle

from PIL import Image
from scipy.misc import imresize, imread
import numpy as np

from ..core.image import *

MEAN_VALUE = np.float32(100.0)

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

dir_path = os.path.dirname(os.path.realpath(__file__))
EIGEN_LIST_FILE = os.path.join(dir_path, 'kitti_eigen_list.txt')


def _file_tuple(file_path):
  """Spilt a valid velodyne/png/oxts data file into multiparts, which have the
  format: (predix, date, seq, device, frame).
  
  Args:
    The data file path name.

  Returns:
    the multipart (predix, date, seq, device, frame) tuple.
  """
  s = file_path.rsplit('/', 5)
  if len(s) != 5 and len(s) != 6:
    raise ValueError("The file path name %s is invalid." % file_path)

  # Without prefix dir name
  if len(s) == 5:
    return '.', s[-5], s[-4], s[3], s[-1].split('.')[0]
  else:
    return s[-6], s[-5], s[-4], s[3], s[-1].split('.')[0]

def _file_path(prefix, date, seq, device, frame):
  """Combine the multipart tuple into file path.

  Args:
    The data path multipart tuple, which has the length of 5 or 6.

  Returns:
    The combined file path
  """

  if device.startswith('image'):
    return os.path.join(prefix, date, seq, device, 'data', frame + '.jpg')
  elif device == 'oxts':
    return os.path.join(prefix, date, seq, device, 'data', frame + '.txt')
  elif device == 'velodyne':
    return os.path.join(prefix, date, seq, device, 'data', frame + '.bin')
  else:
    raise ValueError('The device is not supported.')

def _get_intrinsic(file_path):
    prefix, date, _, device, _ = _file_tuple(file_path)
    calib = read_calib_file(os.path.join(prefix, date, 'calib_cam_to_cam.txt'))
    if device == 'image_02':
        return np.reshape(calib['P_rect_02'], (3, 4))[:3, :3]
    else:
        return np.reshape(calib['P_rect_03'], (3, 4))[:3, :3]

def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(map(float, value.split(' ')))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def generate_depth_map(velo, velo2cam, cam2cam, image_height, 
                       image_width, interp=False, vel_depth=False):
    # load calibration files
    #cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    #velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_02'].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    #velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < image_width) & (velo_pts_im[:,1] < image_height)
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((image_height, image_width))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in collections.Counter(inds).iteritems() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp((image_height, image_width), velo_pts_im)
        return depth, depth_interp
    else:
        return depth


def compute_depth_errors(pred, gt, 
    max_depth=80.0, min_depth=1e-3):

    mask = np.logical_and(gt > min_depth, gt < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
        0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)

    # Scale matching
    scalor = np.median(gt[mask])/np.median(pred[mask])
    pred[mask] *= scalor

    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def _import_from_list(list_file):
    samples = []
    with open(list_file) as f:
        for line in f.readlines():
            s = line.split()
            samples.append(s)
    return samples

class KittiRaw(collections.Sequence):
    def __init__(self, data_dir, image_height, image_width, model_type='stereo',
                 include_list_file=None, exclude_list_file=None, frames_length=1,
                 data_list=['image'], cameras_list=['image_02', 'image_03']):
      if include_list_file != None:
        self.samples = _import_from_list(include_list_file)
      else:
        left = sorted(glob.glob(data_dir + "/*/*/image_02/data/*.jpg"))
        if model_type == 'stereo':
          right = sorted(glob.glob(data_dir + "/*/*/image_03/data/*.jpg"))
          assert len(left) == len(right)
          self.samples = [left, right]
        elif model_type == 'mono':
            right = sorted(glob.glob(data_dir + "/*/*/image_03/data/*.jpg"))
            self.samples = left + right
        else:
            self.samples = left

      self.model_type = model_type
      if exclude_list_file != None:
        self._mask_frames(data_dir, exclude_list_file)
      if model_type == "mono" or model_type == "left_mono":
        self._make_mono_samples(frames_length)
      self.image_height = image_height
      self.image_width = image_width
      self.data_list = data_list

    def _make_mono_samples(self, frames_length):
      samples = []
      length = len(self.samples)
      for i in range(length):
        f = self.samples[i]
        prefix, date, seq, device, fid = _file_tuple(f)
        fid_int = int(fid)
        ok = True
        sample = [f]
        for j in range(1, frames_length):
          frame_path = _file_path(prefix, date, seq, device, "%010d" % (fid_int + j))
          if j + i < length and self.samples[j + i] == frame_path:
            sample.append(frame_path)
          else:
            ok = False
            break
        if ok:
          samples.append(sample)

      self.samples = samples

    def _mask_frames(self, data_dir, exclude_list_file):
      masked_images = set()
      with open(exclude_list_file) as f:
        for item in f.readlines():
          date, seq, frame = item.split()
          masked_images.add(os.path.join(
              data_dir, date, seq, 'image_02', 'data', frame + '.jpg'))
          masked_images.add(os.path.join(
              data_dir, date, seq, 'image_03', 'data', frame + '.jpg'))

      samples = []
      for s in self.samples:
        if self.model_type == 'stereo':
          if not s[0] in masked_images and not s[1] in masked_images:
            samples.append(s)
        else:
          if not s in masked_images:
            samples.append(s)
      self.samples = sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        frames = self.samples[key]
        sample = dict()

        datas = []
        for frame in frames:
            with open(frame, 'rb') as f:
                with Image.open(f) as image:
                    frame_data = image.convert('RGB')
                    frame_width, frame_height = frame_data.size
                    frame_data = imresize(frame_data, (self.image_height, self.image_width))
            datas.append(frame_data)

        sample['image'] = np.concatenate(datas, axis=2)

        if 'intrinsic' in self.data_list:
            intrinsic = _get_intrinsic(frames[0])
            sample['intrinsic'] = scaled_resize_intrinsics(intrinsic, (
                self.image_height / frame_height, 
                self.image_width / frame_width))

        return sample

    # def preprocess(self, inputs):
    #     left_name, right_name, disp_name = inputs
    #     left = imread(left_name, mode='RGB')
    #     right = imread(right_name, mode='RGB')
    #     left = imresize(left, self.target_size, interp='bilinear')
    #     right = imresize(right, self.target_size, interp='bilinear')
    #     left = (left - MEAN_VALUE) / 255
    #     right = (right - MEAN_VALUE) / 255
    #     if self.has_disp:
    #         disp = imread(disp_name, mode='I')
    #         disp = imresize(disp, self.target_size, interp='nearest')
    #         disp = disp[:,:, np.newaxis]
    #         disp = disp * self.target_width / np.float32(disp.shape[1])
    #         return left, right, disp
    #     else:
    #         return left, right


class KITTIStereoFlow(object):
    def __init__(self, data, target_size, has_disp=True):
        if isinstance(data, str):
            self.samples = get_samples(data, has_disp=has_disp)
        else:
            self.samples = data
        self.num_samples = len(self.samples[0])
        self.target_size = target_size
        self.target_height, self.target_width = target_size
        self.has_disp = has_disp
        if has_disp:
            self.type = (np.float32, np.float32, np.float32)
            self.shape = [ target_size + (3,), target_size + (3,), target_size + (1,) ]
        else:
            self.type = (np.float32, np.float32)
            self.shape = [ target_size + (3,), target_size + (3,) ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        return self.samples[key]

    def preprocess(self, inputs):
        left_name, right_name, disp_name = inputs
        left = imread(left_name, mode='RGB')
        right = imread(right_name, mode='RGB')
        left = imresize(left, self.target_size, interp='bilinear')
        right = imresize(right, self.target_size, interp='bilinear')
        left = (left - MEAN_VALUE) / 255
        right = (right - MEAN_VALUE) / 255
        if self.has_disp:
            disp = imread(disp_name, mode='I')
            disp = imresize(disp, self.target_size, interp='nearest')
            disp = disp[:,:, np.newaxis]
            disp = disp * self.target_width / np.float32(disp.shape[1])
            return left, right, disp
        else:
            return left, right


class KittiEigen(object):
    def __init__(self, data_dir, image_height, image_width, 
                 images_list_file=None, mask_min_depth=1e-3, 
                 mask_max_depth=80, data_list=['mono', 'depth_gt']):
        self.image_height = image_height
        self.image_width = image_width
        self.mask_max_depth = mask_max_depth
        self.mask_min_depth = mask_min_depth
        self.data_list = data_list
        self.data_dir = data_dir
        self.samples = _import_from_list(EIGEN_LIST_FILE)
        # if isinstance(data, str):
        #   self.samples = get_samples(data, has_disp=has_disp)
        # else:
        #   self.samples = data
        # self.num_samples = len(self.samples[0])
        # self.target_size = target_size
        # self.target_height, self.target_width = target_size
        # self.has_disp = has_disp
        # if has_disp:
        #   self.type = (np.float32, np.float32, np.float32)
        #   self.shape = [ target_size + (3,), target_size + (3,), target_size + (1,) ]
        # else:
        #   self.type = (np.float32, np.float32)
        #   self.shape = [ target_size + (3,), target_size + (3,) ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        sample = dict()
        date, seq, file_id = self.samples[key]
        if 'mono' in self.data_list:
            image_path = os.path.join(self.data_dir, date, seq, 'image_02', 'data', file_id + '.jpg')
            with Image.open(image_path) as image:
                frame_data = image.convert('RGB')
                frame_width, frame_height = frame_data.size
                frame_data = imresize(frame_data, (self.image_height, self.image_width))
            sample['mono'] = frame_data
        
        if 'depth_gt' in self.data_list or 'depth' in self.data_list:
            velo_path = os.path.join(self.data_dir, date, seq, 
                    'velodyne_points', 'data', file_id + '.bin')
            if not os.path.isfile(velo_path):
                raise IOError("The velodyne file %s not exists." % velo_path)

            velo2cam_path = os.path.join(self.data_dir, date, 
                    'calib_velo_to_cam.txt')
            if not os.path.isfile(velo2cam_path):
                raise IOError("The velodyne to camera calibration file %s not exist." % velo2cam_path)

            cam2cam_path = os.path.join(self.data_dir, date, 'calib_cam_to_cam.txt')
            if not os.path.isfile(cam2cam_path):
                raise IOError("The camera to camera calibration file %s not exist." % cam2cam_path)
            
            velo = load_velodyne_points(velo_path)
            velo2cam = read_calib_file(velo2cam_path)
            cam2cam = read_calib_file(cam2cam_path) 
            if 'depth_gt' in self.data_list:
                depth = generate_depth_map(velo, velo2cam, cam2cam, 
                    frame_height, frame_width, False, True)
                #detph = np.logical_and(depth > self.mask_min_depth, depth < self.mask_min_depth)
                sample['depth_gt'] = depth

            if 'depth' in self.data_list:
                detph = generate_depth_map(velo, velo2cam, cam2cam, 
                    self.image_height, self.image_width, False, True)
                #detph = np.logical_and(depth > self.mask_min_depth, depth < self.mask_min_depth)
                sample['depth'] = depth
        return sample

class KITTIOdometry(object):
    def __init__(self, batch_size, image_height, image_width, gray_dir=None, color_dir=None,
                 velodyne_dir=None, calib_dir=None, gt_dir=None, frames_length=5, seqs_used=None):
        if not gray_dir and not color_dir and not velodyne_dir \
            and not calib_dir and not gt_dir:
            raise ValueError('Must specific a tpye of data.')

        if seqs_used:
            for seq in seqs_used:
                if not seq in range(11):
                    raise ValueError('The sequence id is invalid')
        else:
            seqs_used = range(11)

        self.image_height = image_height
        self.image_width = image_width
        self.seqs_used = seqs_used
        

    def collect_frames(self):
        for seq in  self.seqs_used:
            dir = os.path.join(self.d)









#           gt_files = []
#     gt_calib = []
#     im_sizes = []
#     im_files = []
#     cams = []
#     num_probs = 0
#     for filename in files:
#         filename = filename.split()[0]
#         splits = filename.split('/')
# #         camera_id = filename[-1]   # 2 is left, 3 is right
#         date = splits[0]
#         im_id = splits[4][:10]
#         file_root = '{}/{}'
        
#         im = filename
#         vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)

#         if os.path.isfile(data_root + im):
#             gt_files.append(data_root + vel)
#             gt_calib.append(data_root + date + '/')
#             im_sizes.append(cv2.imread(data_root + im).shape[:2])
#             im_files.append(data_root + im)
#             cams.append(2)
#         else:
#             num_probs += 1
#             print('{} missing'.format(data_root + im))
#     # print(num_probs, 'files missing')

#     return gt_files, gt_calib, im_sizes, im_files, cams


#   def __len__(self):
#       return self.num_samples

#   def __getitem__(self, index):
#       return self.samples[key]

#   def preprocess(self, inputs):
#       left_name, right_name, disp_name = inputs
#       left = imread(left_name, mode='RGB')
#       right = imread(right_name, mode='RGB')
#       left = imresize(left, self.target_size, interp='bilinear')
#       right = imresize(right, self.target_size, interp='bilinear')
#       left = (left - MEAN_VALUE) / 255
#       right = (right - MEAN_VALUE) / 255
#       if self.has_disp:
#           disp = imread(disp_name, mode='I')
#           disp = imresize(disp, self.target_size, interp='nearest')
#           disp = disp[:,:, np.newaxis]
#           disp = disp * self.target_width / np.float32(disp.shape[1])
#           return left, right, disp
#       else:
#           return left, right

def get_samples(data_dir, has_disp=True):
    left = sorted(glob.glob(data_dir + "/image_2/*_10.jpg"))
    right = sorted(glob.glob(data_dir + "/image_3/*_10.jpg"))
    assert len(left) == len(right)
    if has_disp:
        disp = sorted(glob.glob(data_dir + "/disp_occ_0/*_10.jpg"))
        assert len(left) == len(disp)
        return left, right, disp
    else:
        return left, right

def train_and_val(data_dir, target_size, val_ratio=0.15):
    left, right, disp = get_samples(data_dir)
    num_train = int(len(left) * val_ratio)
    train_samples = [ left[:num_train], right[:num_train], disp[:num_train] ]
    val_samples = [ left[num_train:], right[num_train:], disp[num_train:] ]

    return KITTI(train_samples, target_size), KITTI(val_samples, target_size)