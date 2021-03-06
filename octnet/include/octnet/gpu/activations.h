// Copyright (c) 2017, The OctNet authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#ifndef OCTREE_RELU_GPU_H
#define OCTREE_RELU_GPU_H

#include "octnet/core/core.h"

/// Computes the point-wise non-linearity y = max(0,x).
/// @param in  
/// @param inplace indicates if non-linearity should be computed in place,
///                in points to the same memory as out
/// @param out 
OCTREE_API
void octree_relu_gpu(const octree* grid_in, bool inplace, octree* grid_out);

/// Computes the point-wise gradient of y = max(0,x)
/// @param in 
/// @param grad_out
/// @param inplace indicates if non-linearity should be computed in place,
///                in points to the same memory as out
/// @param grad_in
OCTREE_API
void octree_relu_bwd_gpu(const octree* grid_in, const octree* grad_out, bool inplace, octree* grad_in);


/// Computes the point-wise non-linearity y = 1 / (1 + exp(-x)).
/// @param in  
/// @param inplace indicates if non-linearity should be computed in place,
///                in points to the same memory as out
/// @param out 
OCTREE_API
void octree_sigmoid_gpu(const octree* in, bool inplace, octree* out);

/// Computes the point-wise gradient ofy = 1 / (1 + exp(-x)) 
/// @param in 
/// @param grad_out
/// @param inplace indicates if non-linearity should be computed in place,
///                in points to the same memory as out
/// @param grad_in
OCTREE_API
void octree_sigmoid_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in);

/// Computes the log-softmax y = log(exp(x) / sum_c exp(x_c)) over the feature channels
/// @param in
/// @param out
OCTREE_API
void octree_logsoftmax_gpu(const octree* in, octree* out);

/// Computes the gradient of log-softmax y = log(exp(x) / sum_c exp(x_c)) over 
/// the feature channels
/// @param in
/// @param out
OCTREE_API
void octree_logsoftmax_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in);


#endif 
