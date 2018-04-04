-- implementation for OctNet 1 classfication with resolution 8 and mn10

local common = dofile('classification_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local on1_mn40_r8 = {}

function on1_mn40_r8.train(batch_size,vis_files,ll)
  local opt = {}
  local ll = ll or ''
  opt.vx_size = 8
  opt.n_classes = 40
  opt.batch_size = batch_size
  opt.vis_files = vis_files

  opt.ex_data_root = string.format('preprocessed/mn%s/r%s',opt.n_classes,opt.vx_size)
  opt.ex_data_ext = 'oc'
  opt.out_root = string.format('results/on1/mn%s/b%s/s%s/%s',opt.n_classes,opt.batch_size,opt.vx_size,opt.batch_size)

  opt.weightDecay = 0.0001
  opt.learningRate = 1e-3
  opt.n_epochs = 30
  opt.learningRate_steps = {}
  opt.learningRate_steps[15] = 0.1
  opt.optimizer = optim['adam']

  local n_grids = 4096
  opt.net = nn.Sequential()
    :add( oc.VisualOC(skipped) )
    -- conv(1,8)
    :add( oc.OctreeConvolutionMM(1,8, n_grids))
    :add( oc.OctreeReLU(true) )
    :add( oc.VisualOC(skipped) )
    -- conv(8,16)
    :add( oc.OctreeConvolutionMM(8,16, n_grids))
    :add( oc.OctreeReLU(true) )
    :add( oc.VisualOC(skipped) )

    :add( oc.OctreeToCDHW() )
    :add( nn.View(16*8*8*8) )
    -- dropout(0.5)
    :add( nn.Dropout(0.5) )
    -- fc(*,1024)
    :add( nn.Linear(16*8*8*8, 1024) )
    :add( cudnn.ReLU(true) )
    -- fc(1024, classe)
    :add( nn.Linear(1024, opt.n_classes) )

  common.net_he_init(opt.net)
  opt.net:cuda()
  opt.criterion = nn.CrossEntropyCriterion()
  opt.criterion:cuda()
  
  common.classification_worker(opt)
  
  opt = nil
  collectgarbage()
end

return on1_mn40_r8