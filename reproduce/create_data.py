import sys
import numpy as np
import time
import os
from glob import glob
import random
import multiprocessing
import urllib
import zipfile

sys.path.append('../py/')

import pyoctnet

def create_data_with_model(cla=10, vx_res = 64,mn10_path = 'mn10.zip', pad=2, n_rots = 1, n_processes = 1, n_threads = 1, seed_num = 42):

  out_root = './preprocessed/mn' + str(cla) + '/r' + str(vx_res)

  random.seed(seed_num)
  np.random.seed(seed_num)

  # get MN10 data
  if not os.path.exists(mn10_path):
    print('pleas downloading ModelNet10 from "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"')
    exit()

  in_root = '.ignore'
  if not os.path.isdir(in_root):
    print('unzipping ModelNet10')
    mn10 = zipfile.ZipFile(mn10_path, 'r')
    mn10.extractall(in_root)
    mn10.close()

  # create out directory
  if not os.path.isdir(out_root):
    os.makedirs(out_root)

  # list all off files
  off_paths = []
  for root, dirs, files in os.walk(in_root):
    off_paths.extend(glob(os.path.join(root, '*.off')))
  off_paths.sort()

  # fix off header for MN meshes
  print('fixing off headers')
  for path in off_paths:
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    # parse header
    if lines[0].strip().lower() != 'off':
      print(path)
      print(lines[0])

      splits = lines[0][3:].strip().split(' ')
      n_verts = int(splits[0])
      n_faces = int(splits[1])
      n_other = int(splits[2])

      f = open(path, 'w')
      f.write('OFF\n')
      f.write('%d %d %d\n' % (n_verts, n_faces, n_other))
      for line in lines[1:]:
        f.write(line)
      f.close()

  start_t = time.time()
  if n_processes > 1:
    pool = multiprocessing.Pool(processes=n_processes)
  for rot_idx, rot in enumerate(np.linspace(0, 360, n_rots, endpoint=False)):
    rot_out_dir = os.path.join(out_root, 'rot%03d' % np.round(rot))
    if not os.path.isdir(rot_out_dir):
      os.makedirs(rot_out_dir)


  for off_idx, off_path in enumerate(off_paths):
    print('%d pool.apply_async' % off_idx)
    if n_processes > 1:
      pool.apply_async(worker, args=(rot_idx, rot, off_idx, off_path,n_rots,len(off_paths),out_root,vx_res,pad,n_threads,))
    else:
      worker(rot_idx, rot, off_idx, off_path,n_rots,len(off_paths),out_root,vx_res,pad,n_threads)

  if n_processes > 1:
    pool.close()
    pool.join()
  print('create_data took %f[s]' % (time.time() - start_t))



# create grid-octree from off mesh
def worker(rot_idx, rot, off_idx, off_path,n_rots,opsl,out_root,vx_res,pad,n_threads):
  print('%d/%d - %d/%d - %s' % (rot_idx+1, n_rots, off_idx+1, opsl, off_path))

  phi = rot / 180.0 * np.pi
  R = np.array([
      [np.cos(phi), -np.sin(phi), 0],
      [np.sin(phi), np.cos(phi), 0],
      [0, 0, 1]
    ], dtype=np.float32)
  rot_out_dir = os.path.join(out_root, 'rot%03d' % np.round(rot))


  basename, ext = os.path.splitext(os.path.basename(off_path))
  train_test_prefix = os.path.basename(os.path.dirname(off_path))

  print('create octree')
  t = time.time()
  grid = pyoctnet.Octree.create_from_off(off_path.encode(), vx_res,vx_res,vx_res, R, pad=pad, n_threads=n_threads)
  print('  took %f[s]' % (time.time() - t))

  oc_out_path = os.path.join(rot_out_dir, b'%s_%s.oc' % (train_test_prefix.encode(), basename.encode()))
  print('write bin - %s' % oc_out_path)
  t = time.time()
  grid.write_bin(oc_out_path)
  print('  took %f[s]' % (time.time() - t))