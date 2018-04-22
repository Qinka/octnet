local common = require('common')
local dataloader = require('dataloader')

function common.fname_train_test_split_pc(dps, pps)
  local dtrain = {}
  local dtest = {}
  local ptrain = {}
  local ptest = {}

  for i, _ in ipairs(ps) do
    if math.random() < 0.1 then
        table.insert(dtest, dps[i])
        table.insert(ptest, pps[i])
    else
        table.insert(dtrain, dps[i])
        table.insert(ptrain, pps[i])
    end
  end
  return dtrain, dtest, ptrain, ptest
end

function common.seg_worker(opt)
  print(string.format('out_root: %s', opt.out_root))
  -- create out root dir
  paths.mkdir(opt.out_root)

  -- load data_paths
  print('[INFO] load data_paths')
  local t = torch.Timer()
  local data_paths = common.walk_paths_cached(opt.ex_data_root, opt.ex_data_ext)
  local label_paths = common.walk_paths_cached(opt.ex_data_root, opt.ex_data_ext)
  table.sort(data_paths)
  table.sort(label_paths)
  print('[INFO] load data_paths took '..t:time().real..'[s], '..(#data_paths)..', '..(#label_paths))

  print('[INFO] train test split')
  local t = torch.Timer()
  opt.dtr, opt.dtt, opt.ptr, opt.ptt = common.fname_train_test_split_pc(data_paths,label_paths)
  print('[INFO] train test split took '..t:time().real..'[s], '..(#opt.dtr)..', '..(#opt.dtt))

  
  -- get data loaders
  local train_data_loader = dataloader.DataLoader(opt.dtr, opt.ptr, opt.batch_size, opt.parts, opt.vx_size, opt.ex_data_ext, true)
  local test_data_loader = dataloader.DataLoader(opt.dtt, opt.ptt, opt.batch_size, opt.parts, opt.vx_size, opt.ex_data_ext, false)

  -- train
  common.worker(opt, train_data_loader, test_data_loader)
end

return common
