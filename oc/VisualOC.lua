-- Copyright (C) 2018 Johann Lee
-- GPLv3 (temp)

local visdom = require 'visdom'

local force_mask = true
local plot = visdom{server = 'http://localhost', port = 8097}
if not plot:check_connection() then
    print('Could not connect, please ensure the visdom server is running')
    force_mask = false
end

local VisualOC, parent = torch.class('oc.VisualOC', 'oc.OctreeModule')

function VisualOC:__init(label,dense_depth, dense_height, dense_width)
    parent.__init(self)
    self.ok = nil;
    self.video = torch.FloatTensor()
    self.dense_depth = dense_depth
    self.dense_height = dense_height
    self.dense_width = dense_width
    self.label = label or ''
    self.force_mask = force_mask
    self.is_vis = false
end

function VisualOC:enable_vis()
    self.is_vis = self.force_mask
end

function VisualOC:dense_dimensions(octrees)
    if self.dense_depth and self.dense_height and self.dense_width then
        return self.dense_depth, self.dense_height, self.dense_width 
    else
        return octrees:dense_depth(), octrees:dense_height(), octrees:dense_width()
    end
end 

function  VisualOC:updateOutput(input)
    if self.is_vis then
        print('\nsave visual oc\a')
        local dense_depth, dense_height, dense_width = self:dense_dimensions(input)
        local out_size = torch.LongStorage({input:n() *input:feature_size() * dense_depth, 1,dense_height, dense_width})
        self.video:resize(out_size)
        if input._type == 'oc_float' then
            oc.cpu.octree_to_cdhw_cpu(input.grid, dense_depth, dense_height, dense_width, self.video:data())
        elseif input._type == 'oc_cuda' then
            oc.gpu.octree_to_cdhw_gpu(input.grid, dense_depth, dense_height, dense_width, self.video:data())
        end
        self.video:resize(out_size)
        print(self.video:size())
        self.video = self.video - torch.min(self.video)
        self.video = self.video / torch.max(self.video) * 255
        local rt = plot:images{tensor=self.video, opts = {caption = self.label}}
        print(rt)
    end
    self.output = input
    return self.output
end

function VisualOC:updateGradInput(intput,gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end
