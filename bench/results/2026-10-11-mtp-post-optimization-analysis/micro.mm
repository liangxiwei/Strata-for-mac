// Diagnostic A/B: current Strata kernels vs its single-token path / Meld's Q6_K small-batch kernel.
// Inputs to Meld are exactly the dequantized Q8_1 values consumed by Strata.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "ggml-metal-impl.h"

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 3 && argc != 4) return 1;
        const bool check_only = argc == 4;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        id<MTLLibrary> sl = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        id<MTLLibrary> ml = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error];
        if (!sl || !ml) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        auto pipeline = [&](const std::string& name, bool meld) {
            if (pipelines.count(name)) return pipelines[name];
            id<MTLFunction> function;
            if (meld) {
                MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
                const short vals[] = {2, 8, 1, 1, 1};
                for (int i = 0; i < 5; ++i) [cv setConstantValue:&vals[i] type:MTLDataTypeShort atIndex:600+i];
                function = [ml newFunctionWithName:@(name.c_str()) constantValues:cv error:&error];
            } else function = [sl newFunctionWithName:@(name.c_str())];
            if (!function) { fprintf(stderr, "no function %s: %s\n", name.c_str(), error.localizedDescription.UTF8String); exit(2); }
            auto result = [dev newComputePipelineStateWithFunction:function error:&error];
            if (!result) { fprintf(stderr, "pipeline %s: %s\n", name.c_str(), error.localizedDescription.UTF8String); exit(2); }
            pipelines[name] = result;
            return result;
        };
        struct Shape { int type, K, N; };
        printf("type,K,N,mode,round,gpu_us,calls,bitdiff,relative_l2\n");
        for (Shape shape : {Shape{14,2560,248320}, Shape{14,6144,2560},
                            Shape{23,2560,12288}, Shape{20,640,2560}}) {
            const int type=shape.type, K=shape.K, N=shape.N, T=4;
            const int block = type==20 ? 32 : 256;
            const int bytes = type==14 ? 210 : type==23 ? 136 : 18;
            const size_t wbytes = size_t(K/block)*N*bytes;
            std::mt19937 random(type+K+N);
            std::vector<unsigned char> weights(wbytes), xq(size_t(K/32)*36*T);
            for (auto& value : weights) value = random();
            const uint16_t scale = 0x2400;
            for (size_t b=0; b<wbytes; b+=bytes) {
                const unsigned r=random();
                const uint16_t d=check_only?uint16_t(((r>>31)<<15)|((5+(r>>10)%6)<<10)|(r&1023)):scale;
                memcpy(weights.data()+b+(type==14?208:0), &d, 2);
            }
            std::vector<float> xf(size_t(K)*T);
            for (size_t b=0; b<xq.size(); b+=36) {
                const unsigned r=random();
                const uint16_t d=check_only?uint16_t(((8+(r>>10)%6)<<10)|(r&1023)):scale;
                memcpy(xq.data()+b, &d, 2);
                _Float16 d_half;memcpy(&d_half,&d,2);
                int sum=0;
                for (int i=0; i<32; ++i) {
                    const int8_t value = int8_t(random()%255-127);
                    xq[b+4+i]=uint8_t(value); sum+=value;
                    xf[(b/36)*32+i]=float(value)*float(d_half);
                }
                // sum is irrelevant for these formats; a finite value protects accidental reads.
                memcpy(xq.data()+b+2, &scale, 2);
            }
            const int copies=std::min<int>(32,std::max<size_t>(1,((256ull<<20)+wbytes-1)/wbytes));
            std::vector<id<MTLBuffer>> wb;
            for (int i=0;i<copies;++i) wb.push_back([dev newBufferWithBytes:weights.data() length:wbytes options:MTLResourceStorageModeShared]);
            id<MTLBuffer> qx=[dev newBufferWithBytes:xq.data() length:xq.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> fx=[dev newBufferWithBytes:xf.data() length:xf.size()*4 options:MTLResourceStorageModeShared];
            const int reps=std::max<size_t>(8,std::min<size_t>(128,(2ull<<30)/wbytes));
            id<MTLBuffer> yy=[dev newBufferWithLength:size_t(N)*T*reps*4 options:MTLResourceStorageModeShared];
            const std::string fmt=type==14?"q6_k":type==23?"iq4_xs":"iq4_nl";
            const std::string single=type==14?"native_q6_k_mmvq_kernel_large":
                type==23?"native_iq4_xs_direct_sg4":"native_iq4_nl_direct_r4";
            const std::string multi="native_mmvq_multi_kernel_"+fmt+(type==14?"_r1":"_r4");
            pipeline(single,false); pipeline(multi,false);
            if (type==14) pipeline("kernel_mul_mv_ext_q6_K_f32_r1_4",true);
            auto encode=[&](id<MTLComputeCommandEncoder> e,int mode,int copy) {
                const size_t yoffset=size_t(copy%reps)*T*N*4;
                if (mode==2 && type==14) {
                    ggml_metal_kargs_mul_mv_ext args{};
                    args.ne00=K;args.ne01=N;args.ne02=1;args.nb00=bytes;args.nb01=size_t(K/block)*bytes;
                    args.nb02=wbytes;args.nb03=wbytes;args.ne10=K;args.ne11=T;args.ne12=1;
                    args.nb10=4;args.nb11=size_t(K)*4;args.nb12=size_t(K)*T*4;args.nb13=args.nb12;
                    args.ne0=N;args.ne1=T;args.r2=args.r3=1;
                    [e setComputePipelineState:pipeline("kernel_mul_mv_ext_q6_K_f32_r1_4",true)];
                    [e setBytes:&args length:sizeof(args) atIndex:0];
                    [e setBuffer:wb[copy%copies] offset:0 atIndex:1];
                    [e setBuffer:fx offset:0 atIndex:2];
                    [e setBuffer:yy offset:yoffset atIndex:3];
                    [e dispatchThreadgroups:MTLSizeMake((N+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,2,1)];
                } else if (mode==2 || mode==0) {
                    const int nt=mode==0?1:T;
                    const int rows=type==14?1:type==23?16:4;
                    for (int j=0;j<nt;++j) {
                        [e setComputePipelineState:pipeline(single,false)];
                        [e setBuffer:wb[copy%copies] offset:0 atIndex:0];
                        [e setBuffer:qx offset:size_t(j)*(K/32)*36 atIndex:1];
                        [e setBuffer:yy offset:yoffset+size_t(j)*N*4 atIndex:2];
                        [e setBytes:&K length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];
                        [e dispatchThreadgroups:MTLSizeMake((N+rows-1)/rows,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                    }
                } else {
                    const int rows=type==14?1:4, nw=4;
                    [e setComputePipelineState:pipeline(multi,false)];
                    [e setBuffer:wb[copy%copies] offset:0 atIndex:0];[e setBuffer:qx offset:0 atIndex:1];
                    [e setBuffer:yy offset:yoffset atIndex:2];
                    [e setBytes:&K length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];
                    [e setBytes:&T length:4 atIndex:5];[e setBytes:&nw length:4 atIndex:6];
                    [e dispatchThreadgroups:MTLSizeMake((N+rows-1)/rows,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                }
            };
            auto run=[&](int mode,int count) {
                @autoreleasepool {
                    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
                    for(int i=0;i<count;++i) encode(e,mode,i);
                    [e endEncoding];[cb commit];[cb waitUntilCompleted];
                    if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"GPU error %s\n",cb.error.localizedDescription.UTF8String);exit(3);}
                    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/count;
                }
            };
            run(1,1);std::vector<float> ref(size_t(N)*T);memcpy(ref.data(),yy.contents,ref.size()*4);
            long differences=0;double se=0,sr=0;
            run(2,1);const float* got=(const float*)yy.contents;
            for(size_t i=0;i<ref.size();++i){differences+=memcmp(&ref[i],&got[i],4)!=0;se+=double(ref[i]-got[i])*(ref[i]-got[i]);sr+=double(ref[i])*ref[i];}
            const double relative=std::sqrt(se/sr);
            if(!std::isfinite(relative)||(type!=14&&differences)||relative>1e-4){fprintf(stderr,"parity failure %d %d %ld %.9g\n",K,N,differences,relative);return 4;}
            if(check_only){printf("%d,%d,%d,variable_scale_parity4,-1,0,1,%ld,%.9g\n",type,K,N,differences,relative);fflush(stdout);continue;}
            printf("%d,%d,%d,single1,-1,%.3f,1,0,0\n",type,K,N,run(0,reps));fflush(stdout);
            int round=0;
            for(int mode:{1,2,2,1}){
                run(mode,reps);
                auto start=std::chrono::steady_clock::now();std::vector<double> times;
                do {times.push_back(run(mode,reps));} while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10.0);
                std::sort(times.begin(),times.end());
                printf("%d,%d,%d,%s,%d,%.3f,%zu,%ld,%.9g\n",type,K,N,mode==1?"strata_multi4":type==14?"meld_multi4":"strata_separate4",
                       round++,times[times.size()/2],times.size()*reps,mode==1?0:differences,mode==1?0:relative);fflush(stdout);
            }
        }
    }
    return 0;
}
