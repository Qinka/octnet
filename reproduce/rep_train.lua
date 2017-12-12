#!/usr/bin/env th

local start = 3;

if start <=0 then
  local on1_mn10_r8   = dofile('on1_mn10_r8.lua')
  on1_mn10_r8.train(4)
  on1_mn10_r8.train(8)
  on1_mn10_r8.train(16)
  on1_mn10_r8.train(32)
  on1_mn10_r8.train(64)
  on1_mn10_r8.train(128)
  on1_mn10_r8.train(256)
  on1_mn10_r8.train(512)
  on1_mn10_r8   = nil
  collectgarbage()
end

if start <= 1 then
  local on1_mn10_r16  = dofile('on1_mn10_r16.lua')
  on1_mn10_r16.train(4)
  on1_mn10_r16.train(8)
  on1_mn10_r16.train(16)
  on1_mn10_r16.train(32)
  on1_mn10_r16.train(64)
  on1_mn10_r16.train(128)
  on1_mn10_r16.train(256)
  on1_mn10_r16.train(512)
  on1_mn10_r16  = nil
  collectgarbage()
end

if start <= 2 then
  local on1_mn10_r32  = dofile('on1_mn10_r32.lua')
  on1_mn10_r32.train(4)
  on1_mn10_r32.train(8)
  on1_mn10_r32.train(16)
  on1_mn10_r32.train(32)
  on1_mn10_r32.train(64)
  on1_mn10_r32.train(128)
  on1_mn10_r32.train(256)
  on1_mn10_r32  = nil
  collectgarbage()
end

if start <= 3 then
  local on1_mn10_r64  = dofile('on1_mn10_r64.lua')
  on1_mn10_r64.train(4)
  on1_mn10_r64.train(8)
  on1_mn10_r64.train(16)
  on1_mn10_r64.train(32)
  on1_mn10_r64.train(128)
  on1_mn10_r64  = nil
  collectgarbage()
end

if start <= 4 then
  local on1_mn10_r128 = dofile('on1_mn10_r128.lua')
  on1_mn10_r128.train(4)
  on1_mn10_r128.train(8)
  on1_mn10_r128.train(16)
  on1_mn10_r128.train(32)
  on1_mn10_r128 = nil
  collectgarbage()
end

if start <= 5 then
  local on1_mn10_r256 = dofile('on1_mn10_r256.lua')
  on1_mn10_r256.train(4)
  on1_mn10_r256 = nil
  collectgarbage()
end

if start <= 6 then
  local on2_mn10_r8   = dofile('on2_mn10_r8.lua')
  on2_mn10_r8.train(4)
  on2_mn10_r8.train(8)
  on2_mn10_r8.train(16)
  on2_mn10_r8.train(32)
  on2_mn10_r8.train(64)
  on2_mn10_r8.train(128)
  on2_mn10_r8.train(256)
  on2_mn10_r8.train(512)
  on2_mn10_r8   = nil
  collectgarbage()
end

if start <= 7 then
  local on2_mn10_r16  = dofile('on2_mn10_r16.lua')
  on2_mn10_r16.train(4)
  on2_mn10_r16.train(8)
  on2_mn10_r16.train(16)
  on2_mn10_r16.train(32)
  on2_mn10_r16.train(64)
  on2_mn10_r16.train(128)
  on2_mn10_r16.train(256)
  on2_mn10_r16.train(512)
  on2_mn10_r16  = nil
  collectgarbage()
end

if start <= 8 then
  local on2_mn10_r32  = dofile('on2_mn10_r32.lua')
  on2_mn10_r32.train(4)
  on2_mn10_r32.train(8)
  on2_mn10_r32.train(16)
  on2_mn10_r32.train(32)
  on2_mn10_r32.train(64)
  on2_mn10_r32.train(128)
  on2_mn10_r32.train(256)
  on2_mn10_r32  = nil
  collectgarbage()
end

if start <= 9 then
  local on2_mn10_r64  = dofile('on2_mn10_r64.lua')
  on2_mn10_r64.train(4)
  on2_mn10_r64.train(8)
  on2_mn10_r64.train(16)
  on2_mn10_r64.train(32)
  on2_mn10_r64.train(128)
  on2_mn10_r64  = nil
  collectgarbage()
end

if start <= 10 then
  local on2_mn10_r128 = dofile('on2_mn10_r128.lua')
  on2_mn10_r128.train(4)
  on2_mn10_r128.train(8)
  on2_mn10_r128.train(16)
  on2_mn10_r128.train(32)
  on2_mn10_r128 = nil
  collectgarbage()
end

if start <= 11 then
  local on2_mn10_r256 = dofile('on2_mn10_r256.lua')
  on2_mn10_r256.train(4)
  on2_mn10_r256 = nil
  collectgarbage()
end


if start <= 12 then
  local on3_mn10_r8   = dofile('on3_mn10_r8.lua')
  on3_mn10_r8.train(4)
  on3_mn10_r8.train(8)
  on3_mn10_r8.train(16)
  on3_mn10_r8.train(32)
  on3_mn10_r8.train(64)
  on3_mn10_r8.train(128)
  on3_mn10_r8.train(256)
  on3_mn10_r8.train(512)
  on3_mn10_r8   = nil
  collectgarbage()
end

if start <= 13 then
  local on3_mn10_r16  = dofile('on3_mn10_r16.lua')
  on3_mn10_r16.train(4)
  on3_mn10_r16.train(8)
  on3_mn10_r16.train(16)
  on3_mn10_r16.train(32)
  on3_mn10_r16.train(64)
  on3_mn10_r16.train(128)
  on3_mn10_r16.train(256)
  on3_mn10_r16.train(512)
  on3_mn10_r16  = nil
  collectgarbage()
end

if start <= 14 then
  local on3_mn10_r32  = dofile('on3_mn10_r32.lua')
  on3_mn10_r32.train(4)
  on3_mn10_r32.train(8)
  on3_mn10_r32.train(16)
  on3_mn10_r32.train(32)
  on3_mn10_r32.train(64)
  on3_mn10_r32.train(128)
  on3_mn10_r32.train(256)
  on3_mn10_r32  = nil
  collectgarbage()
end

if start <= 15 then
  local on3_mn10_r64  = dofile('on3_mn10_r64.lua')
  on3_mn10_r64.train(4)
  on3_mn10_r64.train(8)
  on3_mn10_r64.train(16)
  on3_mn10_r64.train(32)
  on3_mn10_r64.train(128)
  on3_mn10_r64  = nil
  collectgarbage()
end

if start <= 16 then
  local on3_mn10_r128 = dofile('on3_mn10_r128.lua')
  on3_mn10_r128.train(4)
  on3_mn10_r128.train(8)
  on3_mn10_r128.train(16)
  on3_mn10_r128.train(32)
  on3_mn10_r128 = nil
  collectgarbage()
end

if start <= 17 then
  local on3_mn10_r256 = dofile('on3_mn10_r256.lua')
  on3_mn10_r256.train(4)
  on3_mn10_r256 = nil
  collectgarbage()
end