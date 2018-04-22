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
    out_root = os.path.join('.preprocessed','partseg',str(vx_res))
    
    ## create out directory
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    # list all pts, seg files files
    pts_paths = []
    for root, dirs, files in os.walk(in_root):
        pts_paths.extend(glob(os.path.join(root,'*.pts')))
    pts_paths.sort()

    s_t = time.time()
    pool = None
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

    for p in pts_paths:
        print(p, "transforming")
        if n_processes > 1:
            pool.apply_async(worker,args=(out_root, p,
                                          vx_res,
                                          n_threads,))
        else:
            worker(out_root, p,
                   vx_res,
                   n_threads)
              
    if n_processes > 1:
        pool.close()
        pool.join()
    
    print('create data took %f[s]' % (time.time() - s_t))
    
def worker(outroot: str, filedata : str, vx_res, n_threads=1):
    try:
        k,i = insert_into_pair(filedata)
        prefix,item = os.path.split(filedata)

        kinds = os.listdir(prefix+'_label')
        print('read data', filedata)
        t   = time.time()
        xyz = np.loadtxt(filedata,dtype=np.float32)
        print('\ttook %f[s]' % (time.time() - t))
        
        t = time.time()
        print('read labels')
        ns = []
        os.listdir()
        for k in kinds:
            fp = os.path.join(prefix+'_label',k,i+'.seg')
            if os.path.exists(fp):
                ns.append(np.loadtxt(fp,dtype=np.float32).reshape(-1,1))
            else:
                ns.append(np.zeros((xyz.shape[0],1)))
        plabel = np.concatenate(ns,axis=1) 
        print('\ttook %f[s]' % (time.time() - t), plabel.shape)
        
        print('create octree')
        # object
        grid  = pyoctnet.Octree.create_from_pc_simple(xyz,1,vx_res,vx_res,vx_res,False,n_threads=n_threads)
        # part seg
        label = pyoctnet.Octree.create_from_pc(xyz,plabel,  vx_res,vx_res,vx_res,False,n_threads=n_threads)
        print('\ttook %f[s]' % (time.time() - t))
        
        
        oc_out_path = os.path.join(outroot, k, i + '_' + "pts.oc")
        lb_out_path = os.path.join(outroot, k, i + '_' + "seg.oc")
        if not os.path.exists(os.path.split(oc_out_path)[0]):
            os.makedirs(os.path.split(oc_out_path)[0])
        t = time.time()
        print('write bin', oc_out_path, lb_out_path, grid)
        grid.write_bin(oc_out_path.encode())
        label.write_bin(lb_out_path.encode())
        print('\ttook %f[s]' % (time.time() - t))
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print('error; breake')
        print(e)
        import traceback
        traceback.print_exc()
        return
        
        
def insert_into_pair(file:str):
    dp = file.split('.')
    sp = "".join(dp[0:-1]).split(os.path.sep)
    t = None
    if dp[-1] == 'seg':
        t = sp[-4]
    else:
        t = sp[-3]
    i = sp[-1]
    return (t,i,)


