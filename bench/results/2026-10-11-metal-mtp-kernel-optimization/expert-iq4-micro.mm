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

int main(int argc,char** argv){
 @autoreleasepool{
  const bool bench=argc>2;NSError* error=nil;auto dev=MTLCreateSystemDefaultDevice();
  auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
  if(!lib){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 1;}
  auto queue=[dev newCommandQueue];std::map<int,id<MTLComputePipelineState>> ps;
  for(int r:{0,1,2,4}){std::string n=r?"candidate_resident_iq4_r"+std::to_string(r):"native_resident_down_20";
   auto f=[lib newFunctionWithName:@(n.c_str())];if(!f)return 2;ps[r]=[dev newComputePipelineStateWithFunction:f error:&error];if(!ps[r])return 2;}
  std::mt19937 rng(20261011);long cases=0,checks=0;
  printf("N,K,cap,offsets,variant,round,gpu_us,bitdiff\n");
  for(auto [N,K]:bench?std::vector<std::pair<long,long>>{{2560,640}}:std::vector<std::pair<long,long>>{{513,32},{517,96},{2561,640}})
   for(int cap:bench?std::vector<int>{10,40}:std::vector<int>{3,13,40})
    for(int has=0;has<=(bench?0:1);++has){
     const int ne=bench?512:12,slots=bench?512:10,k=10;const unsigned long row=K/32*18,offset=128,sb=offset+N*row+64;
     std::vector<unsigned char>a(sb*slots),x(K/32*36*cap);for(auto&v:a)v=rng();
     for(int s=0;s<slots;++s)for(long r=0;r<N;++r)for(long b=0;b<K/32;++b){const unsigned v=rng();const uint16_t d=((v>>31)<<15)|((5+(v>>10)%6)<<10)|(v&1023);memcpy(a.data()+s*sb+offset+r*row+b*18,&d,2);}
     for(size_t b=0;b<x.size();b+=36){const unsigned v=rng();const uint16_t d=((8+(v>>10)%6)<<10)|(v&1023);memcpy(x.data()+b,&d,2);memcpy(x.data()+b+2,&d,2);for(int j=0;j<32;++j)x[b+4+j]=rng();}
     const int reps=bench?64:1;std::vector<int>ids(cap*reps),res(ne),perm(ne);std::vector<uint64_t>offs(slots);
     for(int i=0;i<ne;++i){res[i]=i<slots?i:-1;perm[i]=i;}for(int s=0;s<slots;++s)offs[s]=(slots-1-s)*sb;
     for(int i=0;i<cap*reps;++i){if(i%ne==0)std::shuffle(perm.begin(),perm.end(),rng);ids[i]=perm[i%ne];if(!bench&&i%9==4)ids[i]=-1;if(!bench&&i%11==7)ids[i]=ne+3;}
     auto A=[dev newBufferWithBytes:a.data() length:a.size() options:0];auto X=[dev newBufferWithBytes:x.data() length:x.size() options:0];
     auto I=[dev newBufferWithBytes:ids.data() length:ids.size()*4 options:0];auto R=[dev newBufferWithBytes:res.data() length:res.size()*4 options:0];
     auto O=[dev newBufferWithBytes:offs.data() length:offs.size()*8 options:0];auto Y=[dev newBufferWithLength:N*cap*4*reps options:0];
     auto encode=[&](id<MTLComputeCommandEncoder>e,int r,int rep){[e setComputePipelineState:ps[r]];
      [e setBuffer:A offset:0 atIndex:0];[e setBuffer:O offset:0 atIndex:1];[e setBuffer:I offset:rep*cap*4 atIndex:2];[e setBuffer:R offset:0 atIndex:3];[e setBuffer:X offset:0 atIndex:4];
      [e setBytes:&N length:8 atIndex:5];[e setBytes:&K length:8 atIndex:6];[e setBytes:&row length:8 atIndex:7];[e setBytes:&offset length:8 atIndex:8];[e setBytes:&sb length:8 atIndex:9];
      [e setBytes:&ne length:4 atIndex:10];[e setBytes:&k length:4 atIndex:11];[e setBytes:&has length:4 atIndex:12];[e setBuffer:Y offset:rep*N*cap*4 atIndex:13];
      const int rows=r?r:1;[e dispatchThreadgroups:MTLSizeMake((N+8*rows-1)/(8*rows),cap,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];};
     auto run=[&](int r,int count){@autoreleasepool{auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];for(int i=0;i<count;++i)encode(e,r,i);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);exit(3);}return (cb.GPUEndTime-cb.GPUStartTime)*1e6/count;}};
     memset(Y.contents,0xaa,Y.length);run(0,1);std::vector<float>ref(N*cap);memcpy(ref.data(),Y.contents,ref.size()*4);std::map<int,double>screen;
     for(int r:{1,2,4}){memset(Y.contents,0x55,Y.length);run(r,1);long diff=0;auto got=(const float*)Y.contents;for(size_t i=0;i<ref.size();++i)diff+=memcmp(&ref[i],&got[i],4)!=0||!std::isfinite(got[i]);
      ++cases;checks+=ref.size();printf("%ld,%ld,%d,%d,r%d,-1,0,%ld\n",N,K,cap,has,r,diff);fflush(stdout);if(diff)return 4;
      if(bench){std::vector<double>tm;for(int i=0;i<8;++i)tm.push_back(run(r,reps));std::sort(tm.begin(),tm.end());screen[r]=tm[tm.size()/2];printf("%ld,%ld,%d,%d,screen_r%d,-1,%.3f,0\n",N,K,cap,has,r,screen[r]);fflush(stdout);}}
     if(bench){const int best=std::min_element(screen.begin(),screen.end(),[](auto a,auto b){return a.second<b.second;})->first;int round=0;
      for(int r:{0,best,best,0}){run(r,reps);std::vector<double>tm;auto start=std::chrono::steady_clock::now();do{tm.push_back(run(r,reps));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10);std::sort(tm.begin(),tm.end());printf("%ld,%ld,%d,%d,%s,%d,%.3f,0\n",N,K,cap,has,r?("r"+std::to_string(r)).c_str():"old",round++,tm[tm.size()/2]);fflush(stdout);}}
    }
  fprintf(stderr,"Resident IQ4 exact: cases=%ld outputs=%ld differences=0\n",cases,checks);
 }
}
