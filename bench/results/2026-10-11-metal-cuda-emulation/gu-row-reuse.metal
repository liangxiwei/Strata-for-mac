#include "iq_kernels.metal"
// Prototype: preserve load traits and integer scale/division steps, but share
// converted activations across rows and accumulate exact integer float dots.
template<int T> struct GuFloatApply {
    using W=typename iqk_Split<T>::W;
    static float apply(const thread W& w,const thread IqkAct& c) {
        float a=0.0f,b=0.0f;
#pragma unroll
        for(int j=0;j<4;++j)a+=dot(float4(as_type<char4>(w.g[j])),c.a[j]);
#pragma unroll
        for(int j=4;j<8;++j)b+=dot(float4(as_type<char4>(w.g[j])),c.a[j]);
        int sum;
        if constexpr(T==17||T==22) { const int x=(int)a,y=(int)b;sum=(x*w.ls0+y*w.ls1+(x+y)/2)/4; }
        else { sum=(int)(a+b);
            if(T==16)sum=sum*w.ls/8;
            if(T==18)sum=(w.ls*sum+sum/2)/2;
            if(T==21)sum*=w.ls;
        }
        return (w.dw*c.d8)*(float)sum;
    }
};
template<> struct GuFloatApply<23> {
 using W=iqk_Split<23>::W;
 static float apply(const thread W& w,const thread IqkAct& c) {
  float sum=0.0f;
#pragma unroll
  for(int j=0;j<4;++j){sum+=dot(float4(as_type<char4>(w.v[j].x)),c.a[j]);sum+=dot(float4(as_type<char4>(w.v[j].y)),c.a[j+4]);}
  return (w.dw*c.d8)*(float)((int)sum*(w.ls-32));
 }
};
template<int T,int R>
static inline void candidate_gu_rows(constant const uint8_t* arena,constant const ulong* offsets,
 constant const int* ids,constant const int* residency,constant const block_q8_1* xq,
 long n_embd,long n_ff,ulong row_bytes,ulong up_offset,ulong slot_bytes,int n_expert,int k,int has_offsets,
 device float* out,device float* up,uint3 gp,uint tid) {
 using F=iqk_Fmt<T>;
 const int e=(int)gp.y,lane=(int)(tid&31),row0=((int)gp.x*8+(int)(tid>>5))*R;
 if(row0>=2*n_ff)return;
 const int id=ids[e],slot=id>=0&&id<n_expert?residency[id]:-1;
 float sums[R];
#pragma unroll
 for(int r=0;r<R;++r)sums[r]=0.0f;
 if(slot>=0){
  const ulong off=has_offsets?offsets[slot]:(ulong)slot*slot_bytes;
  constant const block_q8_1* xt=xq+(size_t)(e/k)*(n_embd/32);
  for(int kk=lane;kk<(n_embd/F::qk)*F::ipb;kk+=32){
   const int kb=kk/F::ipb,iqs=F::step*(kk%F::ipb);
   const int ab=T==42?iqs:T==23?iqs/4:iqs/2;
   const IqkAct c=iqk_act(xt+kb*(F::qk/32)+ab);
#pragma unroll
   for(int r=0;r<R;++r)if(row0+r<2*n_ff){
    const int row=row0+r;const bool is_up=row>=n_ff;
    constant const uint8_t* wr=arena+off+(is_up?up_offset:0)+(ulong)(is_up?row-n_ff:row)*row_bytes;
    if constexpr(T==42)sums[r]+=iqk_direct_q2_0(wr,kb,iqs,c);
    else {const typename iqk_Split<T>::W w=iqk_Split<T>::load(wr,kb,iqs);sums[r]+=GuFloatApply<T>::apply(w,c);}
   }
  }
 }
#pragma unroll
 for(int r=0;r<R;++r){const float s=slot>=0?iqk_warp_sum(sums[r]):0.0f;const int row=row0+r;
  if(lane==0&&row<2*n_ff){bool iu=row>=n_ff;(iu?up:out)[(size_t)e*n_ff+(iu?row-n_ff:row)]=s;}}
}
#define GU_ROWS(T,R) kernel void candidate_resident_gu_##T##_r##R(IQK_RESIDENT_PARAMS,device float* up [[buffer(14)]],uint3 gp [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { candidate_gu_rows<T,R>(arena,offsets,ids,residency,xq,n_embd,n_ff,row_bytes,weight_offset,slot_bytes,n_expert,k,has_offsets,out,up,gp,tid); }
#define GU_LAYOUTS(T) GU_ROWS(T,2) GU_ROWS(T,4)
GU_LAYOUTS(16) GU_LAYOUTS(17) GU_LAYOUTS(18) GU_LAYOUTS(21) GU_LAYOUTS(22) GU_LAYOUTS(23) GU_LAYOUTS(42)
