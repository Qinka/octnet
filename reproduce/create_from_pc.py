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

    # list all kinds
    all_kinds = os.listdir(in_root)
    items = []
    for k in all_kinds:
        if os.path.isdir(os.path.join(in_root,k)):
            for it in os.listdir(os.path.join(in_root,k,'points')):
                items.append((k,it.split('.')[0],))

    items.sort()

    s_t = time.time()
    pool = None
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

    for it in items:
        print(it, "transforming")
        if n_processes > 1:
            pool.apply_async(worker,args=(in_root, out_root,
                                          it,
                                          vx_res,
                                          n_threads,))
        else:
            worker(in_root, out_root,
                   it,
                   vx_res,
                   n_threads)
              
    if n_processes > 1:
        pool.close()
        pool.join()
    
    print('create data took %f[s]' % (time.time() - s_t))
    
def worker(inroot:str, outroot: str, item, vx_res, n_threads=1):
    try:
        k = item[0]
        i = item[1]
        print(k,i)

        print('read data', item)
        t   = time.time()
        xyz = np.loadtxt(os.path.join(inroot,k,'points',i+'.pts'),dtype=np.float32)
        print('\ttook %f[s]' % (time.time() - t))
        
        t = time.time()
        print('read labels')
        ns = []
        parts = os.listdir(os.path.join(inroot,k,'points_label'))
        for p in parts:
            fp = os.path.join(inroot,k,'points_label',p,i+'.seg')
            if os.path.exists(fp):
                ns.append(np.loadtxt(fp,dtype=np.float32).reshape(-1,1))
            else:
                ns.append(np.zeros((xyz.shape[0],1)))
        plabel = np.concatenate(ns,axis=1) 
        print('\ttook %f[s]' % (time.time() - t), plabel.shape)
        
        print('create octree %d' % vx_res)
        # object
        grid  = pyoctnet.Octree.create_from_pc_simple(xyz,1,    vx_res,vx_res,vx_res,False,n_threads=n_threads)
        # part seg
        label = pyoctnet.Octree.create_from_pc(xyz,plabel,vx_res,vx_res,vx_res,False,n_threads=n_threads)
        print('\ttook %f[s]' % (time.time() - t))
        
        
        oc_out_path = os.path.join(outroot, k, i + '_' + ".pts.oc")
        lb_out_path = os.path.join(outroot, k, i + '_' + ".seg.oc")
        if not os.path.exists(os.path.split(oc_out_path)[0]):
            os.makedirs(os.path.split(oc_out_path)[0])
        t = time.time()
        print('write bin', oc_out_path, lb_out_path, grid)
        grid.write_bin(oc_out_path.encode())
        label.write_to_bin(lb_out_path.encode())
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


