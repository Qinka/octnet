-- Copyright (C) 2018 Johann Lee
-- GPLv3 (temp)

require 'image'
local visdom = require 'visdom'
local first = true;

local plot = visdom{server = 'http://localhost', port = 8097}
if not plot:check_connection() then
    error('Could not connect, please ensure the visdom server is running')
end

local VisualOC, parent = torch.class('oc.VisualOC', 'oc.OctreeModule')

function VisualOC:__init(dense_depth, dense_height, dense_width)
    parent.__init(self)
    self.ok = nil;
    self.video = torch.FloatTensor()
    self.dense_depth = dense_depth
    self.dense_height = dense_height
    self.dense_width = dense_width
end

function VisualOC:dense_dimensions(octrees)
    if self.dense_depth and self.dense_height and self.dense_width then
        return self.dense_depth, self.dense_height, self.dense_width 
    else
        return octrees:dense_depth(), octrees:dense_height(), octrees:dense_width()
    end
end 

function  VisualOC:updateOutput(input)
    if first then
        print('\n\n\n\nsave visual oc\a')
        local dense_depth, dense_height, dense_width = self:dense_dimensions(input)
        local out_size = torch.LongStorage({input:n() * input:feature_size() * dense_depth, dense_height, dense_width})
        self.video:resize(out_size)
        if input._type == 'oc_float' then
            oc.cpu.octree_to_cdhw_cpu(input.grid, dense_depth, dense_height, dense_width, self.video:data())
        elseif input._type == 'oc_cuda' then
            oc.gpu.octree_to_cdhw_gpu(input.grid, dense_depth, dense_height, dense_width, self.video:data())
        end
        self.ok = pcall(plot.image, plot.image, {tensor = video})
        if self.ok then
            print('Uploaded video\n\n\n')
        else
            print('Skipped video\n\n\n')
        end
        plot.image{img=video}
        first = false;
    end
    self.output = input
    return self.output
end

function VisualOC:updateGradInput(intput,gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end
