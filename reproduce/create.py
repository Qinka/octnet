#!env python2

import create_data as c

if __name__ == '__main__':
  c.create_data_with_model(vx_res =   8, mn10_path='../example/01_classification_modelnet/mn10.zip')
  c.create_data_with_model(vx_res =  16, mn10_path='../example/01_classification_modelnet/mn10.zip')
  c.create_data_with_model(vx_res =  32, mn10_path='../example/01_classification_modelnet/mn10.zip')
  c.create_data_with_model(vx_res =  64, mn10_path='../example/01_classification_modelnet/mn10.zip')
  c.create_data_with_model(vx_res = 128, mn10_path='../example/01_classification_modelnet/mn10.zip')
  c.create_data_with_model(vx_res = 256, mn10_path='../example/01_classification_modelnet/mn10.zip')