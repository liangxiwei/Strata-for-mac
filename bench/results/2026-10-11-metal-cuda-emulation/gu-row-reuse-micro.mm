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
int main(int argc,char** argv) { @autoreleasepool {
 if(argc<4)return 1;
 const bool timing=argc>4, sustained=timing&&std::string(argv[4])=="bench";
 auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];NSError* err=nil;
 std::vector<id<MTLLibrary>> libs;
 for(int m=0;m<3;++m){auto l=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[m+1])] error:&err];if(!l){fprintf(stderr,"%s\n",err.localizedDescription.UTF8String);return 2;}libs.push_back(l);}
 std::map<std::pair<int,std::string>,id<MTLComputePipelineState>> pipelines;
 auto pso=[&](int m,const std::string& n){auto key=std::make_pair(m,n);if(!pipelines.count(key)){auto f=[libs[m] newFunctionWithName:@(n.c_str())];if(!f){fprintf(stderr,"missing %s\n",n.c_str());exit(2);}pipelines[key]=[dev newComputePipelineStateWithFunction:f error:&err];if(!pipelines[key])exit(2);}return pipelines[key];};
 std::mt19937 rng(20261011);long checks=0,cases=0;
 printf("type,K,FF,cap,offsets,mode,round,gpu_us,bitdiff\n");
 for(int type:{18,21,22,23,42}) {
  const long qk=type==42?64:256,bytes=type==18?98:type==21?110:type==22?82:type==23?136:18;
  for(auto [K,FF]:timing?std::vector<std::pair<long,long>>{{2560,640}}:std::vector<std::pair<long,long>>{{256,17},{2560,641}})
   for(int cap:timing?std::vector<int>{10,20,40}:std::vector<int>{3,13,40})
    for(int has=0;has<=(timing?0:1);++has) {
     const int ne=timing?512:12,slots=timing?512:10,k=10,reps=timing?32:1;
     const unsigned long row=K/qk*bytes,upoff=FF*row,sb=2*FF*row;
     std::vector<unsigned char>w(sb*slots),x(size_t((cap+9)/10)*K/32*36);
     for(auto&v:w)v=rng();
     for(size_t b=0;b<w.size();b+=bytes){unsigned r=rng();uint16_t d=uint16_t(((r>>31)<<15)|((5+(r>>10)%6)<<10)|(r&1023));memcpy(w.data()+b,&d,2);}
     for(size_t b=0;b<x.size();b+=36){unsigned r=rng();uint16_t d=uint16_t(((8+(r>>10)%6)<<10)|(r&1023));memcpy(x.data()+b,&d,2);memcpy(x.data()+b+2,&d,2);for(int j=0;j<32;++j)x[b+4+j]=rng();}
     std::vector<int>ids(cap*reps),res(ne),perm(ne);std::vector<uint64_t>off(slots);
     for(int i=0;i<ne;++i){res[i]=i<slots?i:-1;perm[i]=i;}for(int i=0;i<slots;++i)off[i]=(slots-1-i)*sb;
     for(int i=0;i<cap*reps;++i){if(i%ne==0)std::shuffle(perm.begin(),perm.end(),rng);ids[i]=perm[i%ne];if(!timing&&i%9==4)ids[i]=-1;if(!timing&&i%11==7)ids[i]=ne+3;}
     auto W=[dev newBufferWithBytes:w.data() length:w.size() options:0];auto X=[dev newBufferWithBytes:x.data() length:x.size() options:0];
     auto I=[dev newBufferWithBytes:ids.data() length:ids.size()*4 options:0];auto R=[dev newBufferWithBytes:res.data() length:res.size()*4 options:0];auto O=[dev newBufferWithBytes:off.data() length:off.size()*8 options:0];
     auto Y=[dev newBufferWithLength:FF*cap*4*reps options:0];auto U=[dev newBufferWithLength:FF*cap*4*reps options:0];
     const std::string name="native_resident_gu_"+std::to_string(type);
     auto run=[&](int m,int calls){@autoreleasepool{auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pso(m,m?"candidate_resident_gu_"+std::to_string(type)+"_r"+std::to_string(m==1?2:4):name)];
      for(int r=0;r<calls;++r){[e setBuffer:W offset:0 atIndex:0];[e setBuffer:O offset:0 atIndex:1];[e setBuffer:I offset:r*cap*4 atIndex:2];[e setBuffer:R offset:0 atIndex:3];[e setBuffer:X offset:0 atIndex:4];
       [e setBytes:&K length:8 atIndex:5];[e setBytes:&FF length:8 atIndex:6];[e setBytes:&row length:8 atIndex:7];[e setBytes:&upoff length:8 atIndex:8];[e setBytes:&sb length:8 atIndex:9];[e setBytes:&ne length:4 atIndex:10];[e setBytes:&k length:4 atIndex:11];[e setBytes:&has length:4 atIndex:12];[e setBuffer:Y offset:r*FF*cap*4 atIndex:13];[e setBuffer:U offset:r*FF*cap*4 atIndex:14];
       [e dispatchThreadgroups:MTLSizeMake((2*FF+8*(m==0?1:m==1?2:4)-1)/(8*(m==0?1:m==1?2:4)),cap,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
      [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);exit(3);}return (cb.GPUEndTime-cb.GPUStartTime)*1e6/calls;}};
     run(0,1);std::vector<float> ref(FF*cap*2);memcpy(ref.data(),Y.contents,FF*cap*4);memcpy(ref.data()+FF*cap,U.contents,FF*cap*4);
     std::map<int,double> screen;
     for(int m:{0,1,2}){run(m,1);long diff=0;for(size_t i=0;i<size_t(FF*cap);++i){diff+=memcmp(&ref[i],(const float*)Y.contents+i,4)!=0;diff+=memcmp(&ref[i+FF*cap],(const float*)U.contents+i,4)!=0;}++cases;checks+=ref.size();printf("%d,%ld,%ld,%d,%d,%d,-1,0,%ld\n",type,K,FF,cap,has,m,diff);fflush(stdout);if(diff)return 4;
      if(timing){run(m,reps);std::vector<double> times;for(int j=0;j<8;++j)times.push_back(run(m,reps));std::sort(times.begin(),times.end());screen[m]=times[times.size()/2];printf("%d,%ld,%ld,%d,%d,%d,-2,%.3f,0\n",type,K,FF,cap,has,m,screen[m]);fflush(stdout);}}
     if(sustained&&((type==18||type==21||type==22)&&cap!=10 || type==42&&cap==10)) {int best=screen[1]<=screen[2]*1.03?1:2;
      if(screen[best]<screen[0]*.92){int round=0;for(int m:{0,best,best,0}){run(m,reps);std::vector<double> times;auto start=std::chrono::steady_clock::now();do{times.push_back(run(m,reps));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10);std::sort(times.begin(),times.end());printf("%d,%ld,%ld,%d,%d,%d,%d,%.3f,0\n",type,K,FF,cap,has,m,round++,times[times.size()/2]);fflush(stdout);}}
     }
    }
 }
 fprintf(stderr,"resident GU cases=%ld outputs=%ld differences=0\n",cases,checks);return 0;
}}
