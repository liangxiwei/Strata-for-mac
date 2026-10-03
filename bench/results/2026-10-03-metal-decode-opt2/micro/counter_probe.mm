#import <Metal/Metal.h>
#include <cstdio>
#include <initializer_list>
#include <unistd.h>
#include <mach/mach_time.h>
int main(){ @autoreleasepool{
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 printf("%s\n",d.name.UTF8String);
 printf("stage %d draw %d dispatch %d blit %d tile %d\n",
  [d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary],
  [d supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary],
  [d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary],
  [d supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary],
  [d supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary]);
 id<MTLCounterSet> ts=nil;
 for(id<MTLCounterSet> cs in d.counterSets){ printf("set %s\n",cs.name.UTF8String); if([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts=cs;}
 for(NSUInteger n: {1024,4096,8192,16384,32768,65536}){
   MTLCounterSampleBufferDescriptor* de=[MTLCounterSampleBufferDescriptor new];
   de.counterSet=ts; de.storageMode=MTLStorageModeShared; de.sampleCount=n;
   NSError* e=nil; id<MTLCounterSampleBuffer> b=[d newCounterSampleBufferWithDescriptor:de error:&e];
   printf("samples %lu -> %s %s\n",(unsigned long)n,b?"ok":"FAIL", e?e.localizedDescription.UTF8String:"");
 }
 MTLTimestamp c0,g0,c1,g1; [d sampleTimestamps:&c0 gpuTimestamp:&g0]; usleep(100000); [d sampleTimestamps:&c1 gpuTimestamp:&g1];
 mach_timebase_info_data_t tb; mach_timebase_info(&tb);
 printf("cpu %llu gpu %llu ratio %.6f timebase %u/%u\n",c1-c0,g1-g0,(double)(c1-c0)/(double)(g1-g0),tb.numer,tb.denom);
}}
