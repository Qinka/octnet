#pragma once
#ifndef _API_H_
#define _API_H_

#ifdef __cplusplus
#define __EXTERN extern "C"
#else
#define __EXTERN 
#endif

#ifndef OCTREE_API
#  ifdef _MSC_VER
#    ifdef OCTREE_EXPORT
#      define OCTREE_API __EXTERN __declspec(dllexport)
#    else
#      define OCTREE_API __EXTERN __declspec(dllimport)
#    endif
#  else
#    define   OCTREE_API __EXTERN
#  endif
#endif

#endif