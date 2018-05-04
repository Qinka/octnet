-- implementation for OctNet 1 classfication with resolution 8 and mn10

local common = dofile('seg_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local seg1_a_r32 = {}

function seg1_a_r32.train(vis_files,ll)
  local opt = {}
  local ll = ll or ''

  opt.vx_size = 256
  opt.parts = 4
  opt.batch_size = 1
  opt.vis_files = vis_files

  opt.ex_data_root = string.format('.preprocessed/partseg/%s/02691156',opt.vx_size)
  opt.ex_data_ext = 'oc'
  opt.ex_label_ext = 'cdwh'
  opt.out_root = string.format('results/seg1/02691156/%s/%s',opt.vx_size,ll)

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
    -- conv(8,8)
    :add( oc.OctreeConvolutionMM(8,8, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    -- conv(8,16)
    :add( oc.OctreeConvolutionMM(8,16, n_grids) )
    :add( oc.OctreeReLU(true) )
    -- conv(16,16)
    :add( oc.OctreeConvolutionMM(16,16, -1) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    -- conv(16,32)
    :add( oc.OctreeConvolutionMM(16,32, n_grids) )
    :add( oc.OctreeReLU(true) )
    -- conv(32,32)
    :add( oc.OctreeConvolutionMM(32,32, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    -- conv(32,64)
    :add( oc.OctreeConvolutionMM(32,64, n_grids) )
    :add( oc.OctreeReLU(true) )
    -- conv(64,64)
    :add( oc.OctreeConvolutionMM(64,64, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridPool2x2x2('max') )
    -- conv(64,128)
    :add( oc.OctreeConvolutionMM(64,128, n_grids) )
    :add( oc.OctreeReLU(true) )
    -- conv(128,128)
    :add( oc.OctreeConvolutionMM(128,128, n_grids) )
    :add( oc.OctreeReLU(true) )
    -- conv(128,128)
    :add( oc.OctreeConvolutionMM(128,128, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridUnpool2x2x2() )
    -- conv(128,64)
    :add( oc.OctreeConvolutionMM(128,64, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridUnpool2x2x2() )
    -- conv(64,32)
    :add( oc.OctreeConvolutionMM(64,32, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridUnpool2x2x2() )
    -- conv(32,16)
    :add( oc.OctreeConvolutionMM(32,16, n_grids) )
    :add( oc.OctreeReLU(true) )
    :add( oc.OctreeGridUnpool2x2x2() )
    -- conv(16,8)
    :add( oc.OctreeConvolutionMM(16,8, n_grids) )
    :add( oc.OctreeReLU(true) )
    -- conv(8,t)
    :add( oc.OctreeConvolutionMM(8,opt.parts, n_grids) )
    :add( oc.OctreeReLU(true) )
--    :add( oc.OctreeToCDHW())

  common.net_he_init(opt.net)
  opt.net:cuda()
  opt.criterion = oc.OctreeCrossEntropyCriterion()
  opt.criterion:cuda()
  common.seg_worker(opt)
  opt = nil
  collectgarbage()
end

return seg1_a_r32
