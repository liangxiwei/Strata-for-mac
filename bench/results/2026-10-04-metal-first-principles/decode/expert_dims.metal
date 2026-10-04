#include "iq_kernels.metal"
kernel void dim_gu_22(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        iqk_resident_body<22, false>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, k, has_offsets, out, up, gp, tid);
    }
kernel void dim_gu_16(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        iqk_resident_body<16, false>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, k, has_offsets, out, up, gp, tid);
    }
kernel void dd2_down(IQK_RESIDENT_PARAMS, uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        iqk_resident_down_direct_q2_0<2>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, has_offsets, out, gp, tid);
    }
kernel void dd4_down(IQK_RESIDENT_PARAMS, uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        iqk_resident_down_direct_q2_0<4>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, has_offsets, out, gp, tid);
    }
kernel void dd8_down(IQK_RESIDENT_PARAMS, uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        iqk_resident_down_direct_q2_0<8>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, has_offsets, out, gp, tid);
    }
kernel void dd16_down(IQK_RESIDENT_PARAMS, uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        iqk_resident_down_direct_q2_0<16>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, has_offsets, out, gp, tid);
    }
