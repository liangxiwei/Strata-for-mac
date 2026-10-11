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

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) return 1;
        const bool bench = argc > 2;
        auto device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
        auto queue = [device newCommandQueue];
        std::map<std::string,id<MTLComputePipelineState>> cache;
        auto pso = [&](const std::string& name) {
            if (!cache.count(name)) {
                auto function = [library newFunctionWithName:@(name.c_str())];
                if (!function) { fprintf(stderr,"missing %s\n",name.c_str());exit(2); }
                cache[name] = [device newComputePipelineStateWithFunction:function error:&error];
                if (!cache[name]) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);exit(2); }
            }
            return cache[name];
        };
        long checked=0,differences=0,cases=0;
        const std::vector<std::pair<int,int>> shapes = bench
            ? std::vector<std::pair<int,int>>{{2560,248320},{6144,2560},{2560,10240}}
            : std::vector<std::pair<int,int>>{{256,513},{512,513},{768,513},{1024,513},{1280,513},{2560,513},{6144,513}};
        printf("K,N,T,variant,round,gpu_us,bitdiff\n");
        for (auto [K,N]:shapes) {
            const int nb=K/256;
            const size_t wbytes=size_t(nb)*N*210;
            std::mt19937 random(K+N);
            std::vector<unsigned char> weights(wbytes),activation(size_t(K/32)*36*8);
            for (auto& v:weights) v=random();
            for (size_t b=0;b<wbytes;b+=210) {
                const unsigned r=random();
                const uint16_t d=uint16_t(((r>>31)<<15)|((5+(r>>10)%6)<<10)|(r&1023));
                memcpy(weights.data()+b+208,&d,2);
            }
            for (size_t b=0;b<activation.size();b+=36) {
                const unsigned r=random();
                const uint16_t d=uint16_t(((8+(r>>10)%6)<<10)|(r&1023));
                memcpy(activation.data()+b,&d,2);memcpy(activation.data()+b+2,&d,2);
                for(int i=0;i<32;++i) activation[b+4+i]=random();
            }
            // A zero activation column also checks zero outputs and scale handling.
            if (!bench) for(size_t b=0;b<size_t(K/32)*36;b+=36) memset(activation.data()+b,0,36);
            const int copies=bench?std::min<int>(32,std::max<size_t>(1,((256ull<<20)+wbytes-1)/wbytes)):1;
            std::vector<id<MTLBuffer>> wb;
            for(int i=0;i<copies;++i) wb.push_back([device newBufferWithBytes:weights.data() length:wbytes options:MTLResourceStorageModeShared]);
            auto xb=[device newBufferWithBytes:activation.data() length:activation.size() options:MTLResourceStorageModeShared];
            const int reps=bench?std::max<size_t>(8,std::min<size_t>(128,(2ull<<30)/wbytes)):1;
            auto yb=[device newBufferWithLength:size_t(N)*8*reps*4 options:MTLResourceStorageModeShared];
            auto encode=[&](id<MTLComputeCommandEncoder> e,int T,int R,int call) {
                const int rows=nb<4?4:1, nw=4;
                const std::string name=R?"candidate_q6_nt"+std::to_string(T)+"_r"+std::to_string(R):
                    T==1?"native_q6_k_mmvq_kernel_"+std::string(nb<4?"small":"large"):
                    "native_mmvq_multi_kernel_q6_k_r"+std::to_string(rows);
                [e setComputePipelineState:pso(name)];
                [e setBuffer:wb[call%copies] offset:0 atIndex:0];[e setBuffer:xb offset:0 atIndex:1];
                [e setBuffer:yb offset:size_t(call%reps)*N*8*4 atIndex:2];
                [e setBytes:&K length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];
                if(!R&&T>1){[e setBytes:&T length:4 atIndex:5];[e setBytes:&nw length:4 atIndex:6];}
                [e dispatchThreadgroups:MTLSizeMake(R?(N+2*R-1)/(2*R):(N+rows-1)/rows,1,1)
                    threadsPerThreadgroup:MTLSizeMake(32,R?2:4,1)];
            };
            auto run=[&](int T,int R,int count) {
                @autoreleasepool {
                    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
                    for(int i=0;i<count;++i)encode(e,T,R,i);
                    [e endEncoding];[cb commit];[cb waitUntilCompleted];
                    if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"GPU %s\n",cb.error.localizedDescription.UTF8String);exit(3);}
                    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/count;
                }
            };
            for(int T=bench?4:1;T<=(bench?4:8);++T) {
                run(T,0,1);std::vector<float> reference(size_t(T)*N);memcpy(reference.data(),yb.contents,reference.size()*4);
                std::map<int,double> screen;
                for(int R:{1,2,4}) {
                    run(T,R,1);const float* got=(const float*)yb.contents;
                    long diff=0;for(size_t i=0;i<reference.size();++i)diff+=memcmp(&reference[i],&got[i],4)!=0||!std::isfinite(got[i]);
                    checked+=reference.size();differences+=diff;++cases;
                    printf("%d,%d,%d,r%d,-1,0,%ld\n",K,N,T,R,diff);fflush(stdout);
                    if(diff)return 4;
                    if(bench){run(T,R,reps);std::vector<double> tm;for(int j=0;j<8;++j)tm.push_back(run(T,R,reps));std::sort(tm.begin(),tm.end());screen[R]=tm[tm.size()/2];
                        printf("%d,%d,%d,screen_r%d,-1,%.3f,0\n",K,N,T,R,screen[R]);fflush(stdout);}
                }
                if(bench) {
                    const int best=std::min_element(screen.begin(),screen.end(),[](auto a,auto b){return a.second<b.second;})->first;
                    int round=0;
                    for(int R:{0,best,best,0}) {
                        run(T,R,reps);std::vector<double> tm;auto start=std::chrono::steady_clock::now();
                        do{tm.push_back(run(T,R,reps));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10.0);
                        std::sort(tm.begin(),tm.end());printf("%d,%d,%d,%s,%d,%.3f,0\n",K,N,T,R?("r"+std::to_string(R)).c_str():"old",round++,tm[tm.size()/2]);fflush(stdout);
                    }
                }
            }
        }
        fprintf(stderr,"Q6 exact: cases=%ld outputs=%ld differences=%ld\n",cases,checked,differences);
        return differences!=0;
    }
}
