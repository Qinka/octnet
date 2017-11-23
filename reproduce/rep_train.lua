#!/usr/bin/env th

local on1_mn10_r8 = dofile('on1_mn10_r8.lua')

on1_mn10_r8.train(4)
on1_mn10_r8.train(16)
on1_mn10_r8.train(32)
on1_mn10_r8.train(64)
on1_mn10_r8.train(128)
on1_mn10_r8.train(256)