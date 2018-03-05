//ot_size_t_buffer* new_ot_size_t_buffer_gpu();
//void free_ot_size_t_buffer_gpu(ot_size_t_buffer* buf);
//void resize_ot_size_t_buffer_gpu(ot_size_t_buffer* buf, ot_size_t N);

// -----------------------------------------------------------------------------
void tree_set_bit_cpu(ot_tree_t* num, int pos);
void tree_unset_bit_cpu(ot_tree_t* num, const int pos);
int tree_n_leafs_cpu(const ot_tree_t* tree);
int tree_data_idx_cpu(const ot_tree_t* tree, const int bit_idx, ot_size_t feature_size);
int octree_mem_capacity_cpu(const octree* grid);
int octree_mem_using_cpu(const octree* grid);

int leaf_idx_to_grid_idx_cpu(const octree* grid, const int leaf_idx);
int data_idx_to_bit_idx_cpu(const ot_tree_t* tree, int data_idx);
int depth_from_bit_idx_cpu(const int bit_idx);
void octree_split_grid_idx_cpu(const octree* in, const int grid_idx, int* n, int* d, int* h, int* w);
void bdhw_from_idx_l1_cpu(const int bit_idx, int* d, int* h, int* w);
void bdhw_from_idx_l2_cpu(const int bit_idx, int* d, int* h, int* w);
void bdhw_from_idx_l3_cpu(const int bit_idx, int* d, int* h, int* w);

ot_tree_t* octree_get_tree_cpu(const octree* grid, ot_size_t grid_idx); 
void octree_clr_trees_cpu(octree* grid_h);
void octree_cpy_trees_cpu_cpu(const octree* src_h, octree* dst_h);
void octree_cpy_prefix_leafs_cpu_cpu(const octree* src_h, octree* dst_h);
void octree_cpy_data_cpu_cpu(const octree* src_h, octree* dst_h);
void octree_copy_cpu(const octree* src, octree* dst);
void octree_upd_n_leafs_cpu(octree* grid_h);
void octree_upd_prefix_leafs_cpu(octree* grid_h);
void octree_fill_data_cpu(octree* grid_h, ot_data_t fill_value);
void octree_cpy_sup_to_sub_cpu(const octree* sup, octree* sub);

void octree_print_cpu(const octree* grid_h);

octree* octree_new_cpu();
void octree_free_cpu(octree* grid_h);

void octree_resize_cpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);
void octree_resize_as_cpu(const octree* src, octree* dst);

void octree_read_header_cpu(const char* path, octree* grid_h);
void octree_read_cpu(const char* path, octree* grid_h);
void octree_write_cpu(const char* path, const octree* grid_h);
void octree_read_batch_cpu(int n_paths, const char** paths, int n_threads, octree* grid_h);
void octree_dhwc_write_cpu(const char* path, const octree* grid_h);
void octree_cdhw_write_cpu(const char* path, const octree* grid_h);
void dense_write_cpu(const char* path, int n_dim, const int* dims, const ot_data_t* data);
ot_data_t* dense_read_cpu(const char* path, int n_dim);
void dense_read_prealloc_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data);
void dense_read_prealloc_batch_cpu(int n_paths, const char** paths, int n_threads, int n_dim, const int* dims, ot_data_t* data); 

void octree_to_dhwc_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_dhwc_bwd_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void octree_to_cdhw_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_cdhw_bwd_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void dhwc_to_octree_sum_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void dhwc_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void dhwc_to_octree_sum_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void dhwc_to_octree_avg_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void cdhw_to_octree_sum_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void cdhw_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void cdhw_to_octree_sum_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void cdhw_to_octree_avg_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void octree_conv3x3x3_sum_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_sum_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_sum_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);
void octree_conv3x3x3_avg_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_avg_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_avg_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);

void octree_pool2x2x2_avg_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_max_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_avg_bwd_cpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);
void octree_pool2x2x2_max_bwd_cpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);

void octree_gridpool2x2x2_avg_cpu(const octree* in, octree* out);
void octree_gridpool2x2x2_max_cpu(const octree* in, octree* out);
void octree_gridpool2x2x2_avg_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridpool2x2x2_max_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);

void octree_gridunpool2x2x2_cpu(const octree* in, octree* out);
void octree_gridunpool2x2x2_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridunpoolguided2x2x2_cpu(const octree* in, const octree* in_struct, octree* out);
void octree_gridunpoolguided2x2x2_bwd_cpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in);

void octree_relu_cpu(const octree* grid_in, bool inplace, octree* grid_out);
void octree_relu_bwd_cpu(const octree* grid_in, const octree* grad_out, bool inplace, octree* grad_in);
void octree_sigmoid_cpu(const octree* in, bool inplace, octree* out);
void octree_sigmoid_bwd_cpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in);
void octree_logsoftmax_cpu(const octree* in, octree* out);
void octree_logsoftmax_bwd_cpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in);

void octree_add_cpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out);
void octree_scalar_mul_cpu(octree* grid, const ot_data_t scalar);
void octree_scalar_add_cpu(octree* grid, const ot_data_t scalar);
void octree_sign_cpu(octree* grid);
void octree_abs_cpu(octree* grid);
void octree_log_cpu(octree* grid);
ot_data_t octree_min_cpu(const octree* grid_in);
ot_data_t octree_max_cpu(const octree* grid_in);

void octree_concat_cpu(const octree* grid_in1, const octree* grid_in2, bool check, octree* grid_out);
void octree_concat_bwd_cpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);
void octree_concat_dense_cpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out);
void octree_concat_dense_bwd_cpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2);

void octree_split_by_prob_cpu(const octree* in, const octree* prob, const ot_data_t thr, bool check, octree* out);
void octree_split_full_cpu(const octree* in, octree* out);
void octree_split_reconstruction_surface_cpu(const octree* in, const octree* rec, ot_data_t rec_thr_from, ot_data_t rec_thr_to, octree* out);
void octree_split_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_split_dense_reconstruction_surface_fres_cpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out);
void octree_split_dense_reconstruction_surface_fres_bwd_cpu(const octree* grad_out, ot_data_t* grad_in);

void octree_combine_n_cpu(const octree** in, const int n, octree* out);
void octree_extract_n_cpu(const octree* in, int from, int to, octree* out);
void octree_mask_by_label_cpu(const octree* labels, int mask_label, bool check, octree* values);
void octree_determine_gt_split_cpu(const octree* struc, const ot_data_t* gt, octree* out);

ot_data_t octree_mse_loss_cpu(const octree* input, const octree* target, bool size_average, bool check);
void octree_mse_loss_bwd_cpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_nll_loss_cpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_nll_loss_bwd_cpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad);

void octree_bce_loss_cpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_loss_bwd_cpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_bce_dense_loss_cpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_dense_loss_bwd_cpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad);

void octree_bce_ds_loss_cpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_ds_loss_bwd_cpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad);

// -----------------------------------------------------------------------------
void volumetric_nn_upsampling_cdhw_cpu(const ot_data_t* in, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* out);
void volumetric_nn_upsampling_cdhw_bwd_cpu(const ot_data_t* grad_out, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* grad_in);

// -----------------------------------------------------------------------------
//THFloatStorage* octree_data_torch_cpu(octree* grid);

// -----------------------------------------------------------------------------
//octree* octree_create_from_dense_cpu(const ot_data_t* data, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads);
