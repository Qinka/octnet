-- Copyright (C) 2018 Johann Lee
-- GPLv3 (temp)

require 'image'
local visdom = require 'visdom'
local first = first;

local plot = visdom{server = 'http://localhost', port = 8097}
if not plot:check_connection() then
   error('Could not connect, please ensure the visdom server is running')
end

local VisualOCModel, parent = torch.class('oc.VisualOC', 'oc.OctreeModule')

function VisualOCModel:__init()
    parent.__init()
    self.ok = nil;
end


function  VisualOCModel:updateOutput(input)
    if first then
        self.ok = pcall(plot.video, plot.video, {tensor = video})
        if self.ok then
            print('Uploaded video')
        else
            print('Skipped video')
        end
        first = false;
    end
    self.output = input
    return output
end

function VisualOCModel:updateGradInput(intput,gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end
