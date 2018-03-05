//#include <cublas_v2.h>
typedef _Bool bool;
typedef int ot_size_t;
typedef int ot_tree_t;
typedef float ot_data_t;
typedef struct cublasContext *cublasHandle_t;

typedef struct {
  ot_size_t* data;
  ot_size_t capacity;
} ot_size_t_buffer;

typedef struct {
  ot_data_t* data;
  ot_size_t capacity;
} ot_data_t_buffer;

typedef struct {
  ot_size_t n;
  ot_size_t grid_depth;
  ot_size_t grid_height;
  ot_size_t grid_width;

  ot_size_t feature_size; 

  ot_size_t n_leafs; 

  ot_tree_t* trees;      
  ot_size_t* prefix_leafs; 
  ot_data_t* data;       

  ot_size_t grid_capacity;
  ot_size_t data_capacity;
} octree;

/////
void *malloc(size_t size);