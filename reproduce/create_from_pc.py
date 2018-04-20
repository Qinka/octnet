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

# shape net core point clouds


def create_oc_frompc(vx_res=256,in_root='PartAnnotation', n_processes=1, n_threads = 1):
    out_root = path.join('.preprocessed','partseg',str(vx_res))
    
    ## create out directory
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    # list all pts, seg files files
    pts_paths = []
    seg_paths = []
    for root, dirs, files in os.walk(in_root):
        pts_paths.extend(glob(os.path.join(root,'*.pts')))
        seg_paths.extend(glob(os.path.join(root,'*.seg')))
    pts_paths.sort()
    seg_paths.sort()
    pairs = {}
    for pp in pts_paths:
        insert_into_pair(pairs,pp)
    for sp in seg_paths:
        insert_into_pair(pairs,sp)

    pool = None
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

    for key in pairs:
        if n_processes > 1:
            pool.applu_async(worker,args=(out_root,
                                          pairs[key]['pts'],
                                          pairs[key]['seg'],
                                          vx_res,
                                          n_thread,))
        else:
            worker(out_root,
                   pairs[key]['pts'],
                   pairs[key]['seg'],
                   vx_res,
                   n_thread)
              
    if n_processes > 1:
        pool.close()
        pool.join()
    
def worker(outroot: str, filedata : str, filelabels : [str], vx_res, n_threads=1):
    print('read data')
    t   = time.time()
    xyz = np.loadtxt(filedata)
    print('\ttook %f[s]' % (time.time() - t))

    t = time.time()
    print('read labels')
    ns = []
    for fl in filelabels:
        ns.append(np.loadtxt(fl))
    plabels = np.concatenate(ns,axis=1) 
    print('\ttook %f[s]' % (time.time() - t))

    print('create octree')
    # object
    grid  = pyoctnet.Octree.create_from_pc_simple(xyz,vx_res,vx_res,vx_res,False,n_threads=n_threads)
    # part seg
    label = pyoctnet.Octree.create_frompc(xyz,plabels,vx_res,vx_res,vx_res,False,n_threads=n_threads)
    print('\ttook %f[s]' % (time.time() - t))
    
    t = time.time()
    print('write bin')
    fprefix = filedata.split['.'][0].split(os.path.sep)
    oc_out_path = os.path.join(outroot,fprefix[0] + os.path.sep + fprefix[2] + '_' + "pts.oc")
    lb_out_path = os.path.join(outroot,fprefix[0] + os.path.sep + fprefix[2] + '_' + "seg.oc")
    grid.write_bin(oc_out_path)
    label.write_bin(lb_out_path)
    print('\ttook %f[s]' % (time.time() - t))

def insert_into_pair(pair,file:str):
    dp = file.split('.')
    sp = "".join(dp[0:-1]).split(os.path.sep)
    t = None
    if dp[-1] == 'seg':
        t = sp[-4]
    else:
        t = sp[-3]
    i = sp[-1]
    if not ((t,i) in pair):
        pair[(t,i)] = {'pts' : [], 'seg' : []}
    if   dp[-1] == 'seg':
        pair[(t,i)]['pts'].append(file)
    elif dp[-1] == 'pts':
        pair[(t,i)]['seg'].append(file)


