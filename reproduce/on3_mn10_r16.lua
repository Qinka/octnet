-- implementation for OctNet 1 classfication with resolution 8 and mn10

local common = dofile('classification_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local on3_mn10_r16 = {}

function on3_mn10_r16.train(batch_size)
  local opt = {}

  opt.vx_size = 16
  opt.n_classes = 10
  opt.batch_size = batch_size

  opt.ex_data_root = string.format('preprocessed/r%s',opt.vx_size)
  opt.ex_data_ext = 'oc'
  opt.out_root = string.format('results/on3/r%s/b%s/%s',opt.vx_size,opt.batch_size,os.time())

  opt.weightDecay = 0.0001
  opt.learningRate = 1e-3
  opt.n_epochs = 20
  opt.learningRate_steps = {}
  opt.learningRate_steps[15] = 0.1
  opt.optimizer = optim['adam']

  local n_grids = 4096
  opt.net = nn.Sequential()
    -- conv(1,8)
    :add( oc.OctreeConvolutionMM(1,8, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )

    :add( oc.OctreeConvolutionMM(8,16, n_grids) )
    :add( oc.OctreeReLU(true) )

    :add( oc.OctreeConvolutionMM(16,16, n_grids) )
    :add( oc.OctreeReLU(true) )
    
    :add( oc.OctreeConvolutionMM(16,16, n_grids) )
    :add( oc.OctreeReLU(true) )

    :add( oc.OctreeToCDHW() )
    :add( nn.View(16*8*8*8) )
    -- dropout(0.5)
    :add( nn.Dropout(0.5) )
    -- fc(*,1024)
    :add( nn.Linear(16*8*8*8, 1024) )
    :add( cudnn.ReLU(true) )
    -- fc(1024,10)
    :add( nn.Linear(1024, opt.n_classes) )

  common.net_he_init(opt.net)
  opt.net:cuda()
  opt.criterion = nn.CrossEntropyCriterion()
  opt.criterion:cuda()

  common.classification_worker(opt)
  
  opt = nil
  collectgarbage()
end

return on3_mn10_r16