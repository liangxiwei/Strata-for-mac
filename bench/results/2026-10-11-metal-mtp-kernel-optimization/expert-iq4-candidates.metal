#include "iq_kernels.metal"

template<int R>
static inline void resident_iq4_nl_direct(constant const uint8_t* arena, constant const ulong* offsets,
    constant const int* ids, constant const int* residency, constant const block_q8_1* xq,
    long n_embd, long n_ff, ulong row_bytes, ulong weight_offset, ulong slot_bytes, int n_expert,
    int has_offsets, device float* out, threadgroup const float* table, uint3 gp, uint tid) {
    const int e=(int)gp.y, lane=(int)(tid&31), row0=((int)gp.x*8+(int)(tid>>5))*R;
    if(row0>=n_embd)return;
    const int id=ids[e], slot=id>=0&&id<n_expert?residency[id]:-1;
    float sum[R];
#pragma unroll
    for(int r=0;r<R;++r)sum[r]=0.0f;
    if(slot>=0){
        const ulong off=has_offsets?offsets[slot]:(ulong)slot*slot_bytes;
        constant const uint8_t* base=arena+off+weight_offset;
        constant const block_q8_1* x=xq+(size_t)e*(n_ff/32);
        for(int kk=lane;kk<(n_ff/32)*2;kk+=32){
            const int kb=kk/2,iqs=2*(kk%2);
            const float d8=iqk_lo2f(x[kb].ds);
            constant const int* aq=reinterpret_cast<constant const int*>(x[kb].qs);
            const float4 a0=float4(as_type<char4>(aq[iqs])),a1=float4(as_type<char4>(aq[iqs+1]));
            const float4 a2=float4(as_type<char4>(aq[iqs+4])),a3=float4(as_type<char4>(aq[iqs+5]));
#pragma unroll
            for(int r=0;r<R;++r){
                constant const block_iq4_nl* w=reinterpret_cast<constant const block_iq4_nl*>(
                    base+(ulong)min(row0+r,(int)n_embd-1)*row_bytes)+kb;
                constant const uint16_t* qw=reinterpret_cast<constant const uint16_t*>(w->qs);
                const uint q0=as_type<uint>(iqk_get_int_b2(qw,iqs)),q1=as_type<uint>(iqk_get_int_b2(qw,iqs+1));
                float v=0.0f;
#pragma unroll
                for(int b=0;b<4;++b){
                    v=fma(table[(q0>>(8*b))&15u],a0[b],v);
                    v=fma(table[(q0>>(8*b+4))&15u],a2[b],v);
                    v=fma(table[(q1>>(8*b))&15u],a1[b],v);
                    v=fma(table[(q1>>(8*b+4))&15u],a3[b],v);
                }
                const float d=(float)w->d*d8;
                sum[r]+=d*(float)(int)v;
            }
        }
    }
#pragma unroll
    for(int r=0;r<R;++r){
        const float v=slot>=0?iqk_warp_sum(sum[r]):0.0f;
        if(lane==0&&row0+r<n_embd)out[(size_t)e*n_embd+row0+r]=v;
    }
}
#define EXP_IQ4(R) \
kernel void candidate_resident_iq4_r##R(IQK_RESIDENT_PARAMS,uint3 gp [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float table[16];if(tid<16)table[tid]=(float)kvalues_iq4nl[tid];threadgroup_barrier(mem_flags::mem_threadgroup); \
    resident_iq4_nl_direct<R>(arena,offsets,ids,residency,xq,n_embd,n_ff,row_bytes,weight_offset,slot_bytes,n_expert,has_offsets,out,table,gp,tid); \
}
EXP_IQ4(1) EXP_IQ4(2) EXP_IQ4(4)
