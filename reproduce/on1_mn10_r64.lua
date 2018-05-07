-- implementation for OctNet 1 classfication with resolution 8 and mn10

local common = dofile('classification_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local on1_mn10_r64 = {}

function on1_mn10_r64.train(cla,batch_size,vis_files,ll)
  local opt = {}
  opt.ll = ll or ''
  opt.vx_size = 64
  opt.n_classes = cla or 10
  opt.batch_size = batch_size
  opt.vis_files = vis_files

  opt.ex_data_root = string.format('preprocessed/%s/mn%s/r%s',opt.ll,opt.n_classes,opt.vx_size)
  opt.ex_data_ext = 'oc'
  opt.out_root = string.format('results/on1/mn%s/r%s/b%s/%s',opt.n_classes,opt.vx_size,opt.batch_size,opt.ll)

  opt.weightDecay = 0.0001
  opt.learningRate = 1e-3
  opt.n_epochs = 20
  opt.learningRate_steps = {}
  opt.learningRate_steps[15] = 0.1
  opt.optimizer = optim['adam']

  local n_grids = 4096
  opt.net = nn.Sequential()
    :add( oc.VisualOC('1') ) 
    -- conv(1,8)
    :add( oc.OctreeConvolutionMM(1,8, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.VisualOC('2') )
    :add( oc.OctreeGridPool2x2x2('max') )

    :add( oc.OctreeConvolutionMM(8,16, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.VisualOC('3') )
    :add( oc.OctreeGridPool2x2x2('max') )

    :add( oc.OctreeConvolutionMM(16,24, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.VisualOC('4') )
    :add( oc.OctreeGridPool2x2x2('max') )

    :add( oc.OctreeConvolutionMM(24,32, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.VisualOC('5') )

    :add( oc.OctreeToCDHW() )
    :add( nn.View(32*8*8*8) )
    -- dropout(0.5)
    :add( nn.Dropout(0.5) )
    -- fc(*,1024)
    :add( nn.Linear(32*8*8*8, 1024) )
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

return on1_mn10_r64
