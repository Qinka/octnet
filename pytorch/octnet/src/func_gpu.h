// -----------------------------------------------------------------------------
//ot_data_t_buffer* new_ot_data_t_buffer_gpu();
//void free_ot_data_t_buffer_gpu(ot_data_t_buffer* buf);
//void resize_ot_data_t_buffer_gpu(ot_data_t_buffer* buf, ot_size_t N);

// -----------------------------------------------------------------------------
//cublasHandle_t octree_torch_current_cublas_handle_gpu(THCState* state);

// -----------------------------------------------------------------------------
void octree_upd_n_leafs_gpu(octree* grid_h);
void octree_upd_prefix_leafs_gpu(octree* grid_h);
void octree_cpy_trees_gpu_gpu(const octree* src_h, octree* dst_h);
void octree_cpy_prefix_leafs_gpu_gpu(const octree* src_h, octree* dst_h);
void octree_copy_gpu(const octree* src, octree* dst);
void octree_fill_data_gpu(octree* grid_h, ot_data_t fill_value);
void octree_cpy_sup_to_sub_gpu(const octree* sup, octree* sub);

octree* octree_new_gpu();
void octree_free_gpu(octree* grid_d);

void octree_to_gpu(const octree* grid_h, octree* grid_d);
void octree_to_cpu(const octree* grid_d, octree* grid_h);

void octree_resize_gpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);
void octree_resize_as_gpu(const octree* src, octree* dst);

void octree_to_dhwc_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_dhwc_bwd_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void octree_to_cdhw_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_cdhw_bwd_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void dhwc_to_octree_sum_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void dhwc_to_octree_avg_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void dhwc_to_octree_sum_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void dhwc_to_octree_avg_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void cdhw_to_octree_sum_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void cdhw_to_octree_avg_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void cdhw_to_octree_sum_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void cdhw_to_octree_avg_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void octree_conv3x3x3_sum_gpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_sum_bwd_gpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_sum_wbwd_gpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);
void octree_conv3x3x3_avg_gpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_avg_bwd_gpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_avg_wbwd_gpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);

void octree_conv_mm_gpu(cublasHandle_t cublas_handle, const octree* grid_in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, int n_grids, octree* grid);
void octree_conv_mm_bwd_gpu(cublasHandle_t cublas_handle, const octree* grad_out, const ot_data_t* weights, int channels_in, int n_grids, octree* grad_in); 
void octree_conv_mm_wbwd_gpu(cublasHandle_t cublas_handle, const octree* in, const octree* grad_out, const float scale, int n_grids, ot_data_t* grad_weights, ot_data_t* grad_bias);

void octree_pool2x2x2_avg_gpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_max_gpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_avg_bwd_gpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);
void octree_pool2x2x2_max_bwd_gpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);

void octree_gridpool2x2x2_avg_gpu(const octree* in, octree* out);
void octree_gridpool2x2x2_max_gpu(const octree* in, octree* out);
void octree_gridpool2x2x2_avg_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridpool2x2x2_max_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);

void octree_gridunpool2x2x2_gpu(const octree* in, octree* out);
void octree_gridunpool2x2x2_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridunpoolguided2x2x2_gpu(const octree* in, const octree* in_struct, octree* out);
void octree_gridunpoolguided2x2x2_bwd_gpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in);

void octree_relu_gpu(const octree* grid_in, bool inplace, octree* grid_out);
void octree_relu_bwd_gpu(const octree* grid_in, const octree* grad_out, bool inplace, octree* grad_in);
void octree_sigmoid_gpu(const octree* in, bool inplace, octree* out);
void octree_sigmoid_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in);
void octree_logsoftmax_gpu(const octree* in, octree* out);
void octree_logsoftmax_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in);

void octree_add_gpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out);
void octree_scalar_mul_gpu(octree* grid, const ot_data_t scalar);
void octree_scalar_add_gpu(octree* grid, const ot_data_t scalar);
void octree_sign_gpu(octree* grid);
void octree_abs_gpu(octree* grid);
void octree_log_gpu(octree* grid);
ot_data_t octree_min_gpu(const octree* grid_in);
ot_data_t octree_max_gpu(const octree* grid_in);

void octree_concat_gpu(const octree* grid_in1, const octree* grid_in2, bool check, octree* grid_out);
void octree_concat_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);
void octree_concat_ds_gpu(const octree* in1, const octree* in2, octree* out);
void octree_concat_ds_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);
void octree_concat_dense_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out);
void octree_concat_dense_bwd_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2);

void octree_split_by_prob_gpu(const octree* in, const octree* prob, const ot_data_t thr, bool check, octree* out);
void octree_split_full_gpu(const octree* in, octree* out);
void octree_split_reconstruction_surface_gpu(const octree* in, const octree* rec, ot_data_t rec_thr_from, ot_data_t rec_thr_to, octree* out);
void octree_split_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_split_dense_reconstruction_surface_gpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int structure_type, octree* out);
void octree_split_dense_reconstruction_surface_bwd_gpu(const octree* grad_out, ot_data_t* grad_in);
void octree_split_dense_reconstruction_surface_fres_gpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out);
void octree_split_dense_reconstruction_surface_fres_bwd_gpu(const octree* grad_out, ot_data_t* grad_in);
void octree_split_tsdf_gpu(const ot_data_t* features, const ot_data_t* reconstruction, const octree* guide, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int band, octree* out);

void octree_mask_by_label_gpu(const octree* labels, int mask_label, bool check, octree* values);
void octree_determine_gt_split_gpu(const octree* struc, const ot_data_t* gt, octree* out);

ot_data_t octree_mse_loss_gpu(const octree* input, const octree* target, bool size_average, bool check);
void octree_mse_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);
ot_data_t octree_mse_ds_loss_gpu(const octree* input, const octree* target, bool size_average);
void octree_mse_loss_ds_bwd_gpu(const octree* input, const octree* target, bool size_average, octree* grad);

void octree_nll_loss_gpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_nll_loss_bwd_gpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad);

void octree_bce_loss_gpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_bce_dense_loss_gpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_dense_loss_bwd_gpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad);

void octree_bce_ds_loss_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_ds_loss_bwd_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad);

void dense_bce_loss_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t* output, ot_data_t* total_weight);
void dense_bce_loss_bwd_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t total_weight, ot_data_t* grad); 

// -----------------------------------------------------------------------------
void volumetric_nn_upsampling_cdhw_gpu(const ot_data_t* in, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* out);
void volumetric_nn_upsampling_cdhw_bwd_gpu(const ot_data_t* grad_out, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* grad_in);

