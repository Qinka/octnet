pcloader = {}

local PCLoader = torch.class('dataloader.DataLoader')

function PCLoader:__init(data_paths, label_paths, batch_size, parts, vx_size, ex_data_ext, full_batches)
  self.data_paths = data_paths or error('')
  self.label_paths= label_paths or error('')
  self.labels = labels or error('')
  self.batch_size = batch_size or error('')
  self.vx_size = vx_size or error('')
  self.ex_data_ext = ex_data_ext or error('')
  self.full_batches = full_batches or false
  self.parts = parts or error('')

  assert(#self.data_paths == self.labels:size(1))
  self.n_samples = self.labels:size(1)

  self.data_idx = 0
end

function PCLoader:getBatch()
  local bs = math.min(self.batch_size, #self.data_paths - self.data_idx)

  if self.data_idx == 0 then
    local cnt = #self.data_paths
    while cnt > 1 do 
      local idx = math.random(cnt)

      local tmp = self.data_paths[idx]
      self.data_paths[idx] = self.data_paths[cnt]
      self.data_paths[cnt] = tmp

      local tmp = self.label_paths[idx]
      self.label_paths[idx] = self.label_paths[cnt]
      self.label_paths[cnt] = tmp

      cnt = cnt - 1
    end
  end

  local used_paths = {}
  for batch_idx = 1, bs do
    self.data_idx = self.data_idx + 1
    local used_path = self.data_paths[self.data_idx]

    table.insert(used_paths, used_path)
  end

  if self.ex_data_ext == 'cdhw' then
    self.data_cpu = torch.FloatTensor(bs, 1, self.vx_size, self.vx_size, self.vx_size)
    oc.read_dense_from_bin_batch(used_paths, self.data_cpu)
    self.data_gpu = self.data_gpu or torch.CudaTensor()
    self.data_gpu:resize(self.data_cpu:size())
    self.data_gpu:copy(self.data_cpu)
  elseif self.ex_data_ext == 'oc' then 
    self.data_cpu = oc.FloatOctree()
    self.data_cpu:read_from_bin_batch(used_paths)
    self.data_gpu = self.data_cpu:cuda(self.data_gpu)
  else
    error('unknown ex_data_ext: '..self.ex_data_ext)
  end 

  if self.ex_data_ext == 'cdhw' then
    self.label_cpu = torch.FloatTensor(bs, self.parts, self.vx_size, self.vx_size, self.vx_size)
    oc.read_dense_from_bin_batch(used_paths, self.label_cpu)
    self.label_gpu = self.label_gpu or torch.CudaTensor()
    self.label_gpu:resize(self.label_cpu:size())
    self.label_gpu:copy(self.label_cpu)
  elseif self.ex_data_ext == 'oc' then 
    self.label_cpu = oc.FloatOctree()
    self.label_cpu:read_from_bin_batch(used_paths)
    self.label_gpu = self.label_cpu:cuda(self.label_gpu)
  else
    error('unknown ex_data_ext: '..self.ex_data_ext)
  end 

  if (self.full_batches and (#self.data_paths - self.data_idx) < self.batch_size) or 
     (not self.full_batches and self.data_idx >= #self.data_paths) then 
    self.data_idx = 0 
  end

  collectgarbage(); collectgarbage()

  return self.data_gpu, self.label_gpu
end


function PCLoader:size()
  return self.n_samples
end

function PCLoader:n_batches()
  if self.full_batches then
    return math.floor(self.n_samples / self.batch_size)
  else
    return math.ceil(self.n_samples / self.batch_size)
  end
end


function PCLoader:getFiles(fn)
  if fn then
    if type(fn) ~= "table" then
      fn = {fn}
    end
    if self.ex_data_ext == 'cdhw' then
    self.data_cpu = torch.FloatTensor(table.getn(fn), 1, self.vx_size, self.vx_size, self.vx_size)
    oc.read_dense_from_bin_batch(fn, self.data_cpu)
    self.data_gpu = self.data_gpu or torch.CudaTensor()
    self.data_gpu:resize(self.data_cpu:size())
    self.data_gpu:copy(self.data_cpu)
  elseif self.ex_data_ext == 'oc' then 
    self.data_cpu = oc.FloatOctree()
    self.data_cpu:read_from_bin_batch(fn)
    self.data_gpu = self.data_cpu:cuda(self.data_gpu)
  else
    error('unknown ex_data_ext: '..self.ex_data_ext)
  end 
    collectgarbage(); collectgarbage()
    return self.data_gpu
  else
    error(fn, "wrong file")
  end
end

return pcloader
