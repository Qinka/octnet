-- implementation for OctNet 1 classfication with resolution 8 and mn10

local common = dofile('classification_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local on2_mn10_r256 = {}

function on2_mn10_r256.train(batch_size,skipped,ll)
  local opt = {}
  local ll = ll or ''
  opt.vis_skipped = skipped
  opt.vx_size = 256
  opt.n_classes = 10
  opt.batch_size = batch_size

  opt.ex_data_root = string.format('preprocessed/mn%s/r%s',opt.n_classes,opt.vx_size)
  opt.ex_data_ext = 'oc'
  opt.out_root = string.format('results/on2/mn%s/r%s/b%s/%s',opt.n_classes,opt.vx_size,opt.batch_size,ll)

  opt.weightDecay = 0.0001
  opt.learningRate = 1e-3
  opt.n_epochs = 20
  opt.learningRate_steps = {}
  opt.learningRate_steps[15] = 0.1
  opt.optimizer = optim['adam']

  local n_grids = 4096
  opt.net = nn.Sequential()
    :add( oc.VisualOC('l0') )

    -- conv(1,8)
    :add( oc.OctreeConvolutionMM(1,8, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    
    :add( oc.VisualOC('l1') )

    :add( oc.OctreeConvolutionMM(8,16, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    
    :add( oc.VisualOC('l2') )

    :add( oc.OctreeConvolutionMM(16,24, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    
    :add( oc.VisualOC('l3') )
    
    :add( oc.OctreeConvolutionMM(24,32, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    
    :add( oc.VisualOC('l4') )

    :add( oc.OctreeConvolutionMM(32,40, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    
    :add( oc.VisualOC('l5') )

    :add( oc.OctreeConvolutionMM(40,48, n_grids) )
    :add( oc.OctreeReLU(true) )
    
    :add( oc.VisualOC('l6') )

    :add( oc.OctreeConvolutionMM(48,48, n_grids) )
    :add( oc.OctreeReLU(true) )
    
    :add( oc.VisualOC('l7') )

    :add( oc.OctreeToCDHW() )
    :add( nn.View(48*8*8*8) )
    -- dropout(0.5)
    :add( nn.Dropout(0.5) )
    -- fc(*,1024)
    :add( nn.Linear(48*8*8*8, 1024) )
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

return on2_mn10_r256