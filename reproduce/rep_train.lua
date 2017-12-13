#!/usr/bin/env th

local start = 3;

local on1_mn10_r8   = dofile('on1_mn10_r8.lua')
if start <= 80041 then
  on1_mn10_r8.train(4)
end
if start <= 80081 then
  on1_mn10_r8.train(8)
end
if start <= 80161 then
  on1_mn10_r8.train(16)
end
if start <= 80321 then
  on1_mn10_r8.train(32)
end
if start <= 80641 then
  on1_mn10_r8.train(64)
end
if start <= 81281 then
  on1_mn10_r8.train(128)
end
if start <= 82561 then
  on1_mn10_r8.train(256)
end
if start <= 85121 then
  on1_mn10_r8.train(512)
end
on1_mn10_r8   = nil
collectgarbage()

local on2_mn10_r8   = dofile('on2_mn10_r8.lua')
if start <= 80042 then
  on2_mn10_r8.train(4)
end
if start <= 80082 then
  on2_mn10_r8.train(8)
end
if start <= 80162 then
  on2_mn10_r8.train(16)
end
if start <= 80322 then
  on2_mn10_r8.train(32)
end
if start <= 80642 then
  on2_mn10_r8.train(64)
end
if start <= 81282 then
  on2_mn10_r8.train(128)
end
if start <= 82562 then
  on2_mn10_r8.train(256)
end
if start <= 85122 then
  on2_mn10_r8.train(512)
end
on2_mn10_r8   = nil
collectgarbage()

local on3_mn10_r8   = dofile('on3_mn10_r8.lua')
if start <= 80043 then
  on3_mn10_r8.train(4)
end
if start <= 80083 then
  on3_mn10_r8.train(8)
end
if start <= 80163 then
  on3_mn10_r8.train(16)
end
if start <= 80323 then
  on3_mn10_r8.train(32)
end
if start <= 80643 then
  on3_mn10_r8.train(64)
end
if start <= 81283 then
  on3_mn10_r8.train(128)
end
if start <= 82563 then
  on3_mn10_r8.train(256)
end
if start <= 85123 then
  on3_mn10_r8.train(512)
end
on3_mn10_r8   = nil
collectgarbage()

local on1_mn10_r16  = dofile('on1_mn10_r16.lua')
if start <= 160041 then
  on1_mn10_r16.train(4)
end
if start <= 160081 then
  on1_mn10_r16.train(8)
end
if start <= 160161 then
  on1_mn10_r16.train(16)
end
if start <= 160321 then
  on1_mn10_r16.train(32)
end
if start <= 160641 then
  on1_mn10_r16.train(64)
end
if start <= 161281 then
  on1_mn10_r16.train(128)
end
if start <= 162561 then
  on1_mn10_r16.train(256)
end
if start <= 165121 then
  on1_mn10_r16.train(512)
end
on1_mn10_r16  = nil
collectgarbage()

local on2_mn10_r16  = dofile('on2_mn10_r16.lua')
if start <= 160042 then
  on2_mn10_r16.train(4)
end
if start <= 160082 then
  on2_mn10_r16.train(8)
end
if start <= 160162 then
  on2_mn10_r16.train(16)
end
if start <= 160322 then
  on2_mn10_r16.train(32)
end
if start <= 160642 then
  on2_mn10_r16.train(64)
end
if start <= 161282 then
  on2_mn10_r16.train(128)
end
if start <= 162562 then
  on2_mn10_r16.train(256)
end
if start <= 165122 then
  on2_mn10_r16.train(512)
end
on2_mn10_r16  = nil
collectgarbage()

local on3_mn10_r16  = dofile('on3_mn10_r16.lua')
if start <= 160043 then
  on3_mn10_r16.train(4)
end
if start <= 160083 then
  on3_mn10_r16.train(8)
end
if start <= 160163 then
  on3_mn10_r16.train(16)
end
if start <= 160323 then
  on3_mn10_r16.train(32)
end
if start <= 160543 then
  on3_mn10_r16.train(64)
end
if start <= 161283 then
  on3_mn10_r16.train(128)
end
if start <= 162563 then
  on3_mn10_r16.train(256)
end
if start <= 165123 then
  on3_mn10_r16.train(512)
end
on3_mn10_r16  = nil
collectgarbage()

local on1_mn10_r32  = dofile('on1_mn10_r32.lua')
if start <= 320041 then
  on1_mn10_r32.train(4)
end
if start <= 320081 then
  on1_mn10_r32.train(8)
end
if start <= 320161 then
  on1_mn10_r32.train(16)
end
if start <= 320321 then
  on1_mn10_r32.train(32)
end
if start <= 320641 then
  on1_mn10_r32.train(64)
end
if start <= 321281 then
  on1_mn10_r32.train(128)
end
if start <= 322561 then
  on1_mn10_r32.train(256)
end
on1_mn10_r32  = nil
collectgarbage()

local on2_mn10_r32  = dofile('on2_mn10_r32.lua')
if start <= 320042 then
  on2_mn10_r32.train(4)
end
if start <= 320082 then
  on2_mn10_r32.train(8)
end
if start <= 320162 then
  on2_mn10_r32.train(16)
end
if start <= 320322 then
  on2_mn10_r32.train(32)
end
if start <= 320642 then
  on2_mn10_r32.train(64)
end
if start <= 321282 then
  on2_mn10_r32.train(128)
end
if start <= 322562 then
  on2_mn10_r32.train(256)
end
on2_mn10_r32  = nil
collectgarbage()

local on3_mn10_r32  = dofile('on3_mn10_r32.lua')
if start <= 320043 then
  on3_mn10_r32.train(4)
end
if start <= 320083 then
  on3_mn10_r32.train(8)
end
if start <= 320163 then
  on3_mn10_r32.train(16)
end
if start <= 320323 then
  on3_mn10_r32.train(32)
end
if start <= 320643 then
  on3_mn10_r32.train(64)
end
if start <= 321283 then
  on3_mn10_r32.train(128)
end
if start <= 322563 then
  on3_mn10_r32.train(256)
end
on3_mn10_r32  = nil
collectgarbage()

local on1_mn10_r64  = dofile('on1_mn10_r64.lua')
if start <= 640041 then
  on1_mn10_r64.train(4)
end
if start <= 640081 then
  on1_mn10_r64.train(8)
end
if start <= 640161 then
  on1_mn10_r64.train(16)
end
if start <= 640321 then
  on1_mn10_r64.train(32)
end
if start <= 641281 then
  on1_mn10_r64.train(128)
end
on1_mn10_r64  = nil
collectgarbage()

local on2_mn10_r64  = dofile('on2_mn10_r64.lua')
if start <= 640042 then
  on2_mn10_r64.train(4)
end
if start <= 640082 then
  on2_mn10_r64.train(8)
end
if start <= 640162 then
  on2_mn10_r64.train(16)
end
if start <= 640322 then
  on2_mn10_r64.train(32)
end
if start <= 640642 then
  on2_mn10_r64.train(64)
end
if start <= 641282 then
  on2_mn10_r64.train(128)
end
on2_mn10_r64  = nil
collectgarbage()

local on3_mn10_r64  = dofile('on3_mn10_r64.lua')
if start <= 640043 then
  on3_mn10_r64.train(4)
end
if start <= 640083 then
  on3_mn10_r64.train(8)
end
if start <= 640163 then
  on3_mn10_r64.train(16)
end
if start <= 640323 then
  on3_mn10_r64.train(32)
end
if start <= 640643 then
  on3_mn10_r64.train(64)
end
if start <= 641283 then
  on3_mn10_r64.train(128)
end
on3_mn10_r64  = nil
collectgarbage()

local on1_mn10_r128 = dofile('on1_mn10_r128.lua')
if start <= 1280041 then
  on1_mn10_r128.train(4)
end
if start <= 1280081 then
  on1_mn10_r128.train(8)
end
if start <= 1280161 then
  on1_mn10_r128.train(16)
end
if start <= 1280321 then
  on1_mn10_r128.train(32)
end
on1_mn10_r128 = nil
collectgarbage()

local on2_mn10_r128 = dofile('on2_mn10_r128.lua')
if start <= 1280042 then
  on2_mn10_r128.train(4)
end
if start <= 1280082 then
  on2_mn10_r128.train(8)
end
if start <= 1280162 then
  on2_mn10_r128.train(16)
end
if start <= 1280322 then
  on2_mn10_r128.train(32)
end
on2_mn10_r128 = nil
collectgarbage()

local on3_mn10_r128 = dofile('on3_mn10_r128.lua')
if start <= 1280043 then
  on3_mn10_r128.train(4)
end
if start <= 1280083 then
  on3_mn10_r128.train(8)
end
if start <= 1280163 then
  on3_mn10_r128.train(16)
end
if start <= 1280323 then
  on3_mn10_r128.train(32)
end
on3_mn10_r128 = nil
collectgarbage()


local on1_mn10_r256 = dofile('on1_mn10_r256.lua')
if start <= 2560041 then
  on1_mn10_r256.train(4)
end
on1_mn10_r256 = nil
collectgarbage()

local on2_mn10_r256 = dofile('on2_mn10_r256.lua')
if start <= 2560042 then
  on2_mn10_r256.train(4)
end
on2_mn10_r256 = nil
collectgarbage()

local on3_mn10_r256 = dofile('on3_mn10_r256.lua')
if start <= 2560043 then
  on3_mn10_r256.train(4)
end
on3_mn10_r256 = nil
collectgarbage()