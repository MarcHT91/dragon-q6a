#include <stdio.h>
#include <string.h>
#include "egl_min.h"
#include <drm_fourcc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_drm.h>
static enum AVPixelFormat gf(AVCodecContext*c,const enum AVPixelFormat*p){for(;*p!=-1;p++)if(*p==AV_PIX_FMT_DRM_PRIME)return *p;return AV_PIX_FMT_NONE;}
static EGLDisplay d; static PFNEGLCREATEIMAGEKHRPROC ci; static PFNEGLDESTROYIMAGEKHRPROC di;
static void try_import(const AVDRMFrameDescriptor*D,int w,int h,int usemod,int pl_limit){
  const AVDRMLayerDescriptor*L=&D->layers[0]; EGLint a[64]; int n=0;
  a[n++]=EGL_WIDTH;a[n++]=w;a[n++]=EGL_HEIGHT;a[n++]=h;a[n++]=EGL_LINUX_DRM_FOURCC_EXT;a[n++]=L->format;
  static const EGLint fd_[]={EGL_DMA_BUF_PLANE0_FD_EXT,EGL_DMA_BUF_PLANE1_FD_EXT,EGL_DMA_BUF_PLANE2_FD_EXT};
  static const EGLint of_[]={EGL_DMA_BUF_PLANE0_OFFSET_EXT,EGL_DMA_BUF_PLANE1_OFFSET_EXT,EGL_DMA_BUF_PLANE2_OFFSET_EXT};
  static const EGLint pi_[]={EGL_DMA_BUF_PLANE0_PITCH_EXT,EGL_DMA_BUF_PLANE1_PITCH_EXT,EGL_DMA_BUF_PLANE2_PITCH_EXT};
  static const EGLint ml[]={EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT};
  static const EGLint mh[]={EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT};
  for(int i=0;i<L->nb_planes&&i<pl_limit;i++){const AVDRMObjectDescriptor*o=&D->objects[L->planes[i].object_index];
    a[n++]=fd_[i];a[n++]=o->fd;a[n++]=of_[i];a[n++]=L->planes[i].offset;a[n++]=pi_[i];a[n++]=L->planes[i].pitch;
    if(usemod){a[n++]=ml[i];a[n++]=o->format_modifier&0xffffffff;a[n++]=mh[i];a[n++]=o->format_modifier>>32;}}
  a[n++]=EGL_NONE;
  EGLImageKHR im=ci(d,EGL_NO_CONTEXT,EGL_LINUX_DMA_BUF_EXT,NULL,a);
  printf("  import usemod=%d -> %s (err 0x%x)\n",usemod,im?"OK":"FAIL",eglGetError()); if(im)di(d,im);
}
int main(int c,char**v){
  PFNEGLGETPLATFORMDISPLAYEXTPROC gp=(void*)eglGetProcAddress("eglGetPlatformDisplayEXT");
  d=gp(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,NULL); eglInitialize(d,0,0);
  ci=(void*)eglGetProcAddress("eglCreateImageKHR"); di=(void*)eglGetProcAddress("eglDestroyImageKHR");
  PFNEGLQUERYDMABUFMODIFIERSEXTPROC qm=(void*)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
  EGLuint64KHR mods[32];EGLBoolean ext[32];EGLint nm=0; qm(d,DRM_FORMAT_NV12,32,mods,ext,&nm);
  printf("EGL NV12 modifiers:");for(int i=0;i<nm;i++)printf(" %llx%s",(unsigned long long)mods[i],ext[i]?"(ext-only)":"");puts("");
  AVFormatContext*fc=NULL; avformat_open_input(&fc,v[1],0,0); avformat_find_stream_info(fc,0);
  int si=av_find_best_stream(fc,AVMEDIA_TYPE_VIDEO,-1,-1,0,0);
  const AVCodec*cd=avcodec_find_decoder_by_name(v[2]); AVCodecContext*cx=avcodec_alloc_context3(cd);
  avcodec_parameters_to_context(cx,fc->streams[si]->codecpar); cx->get_format=gf; avcodec_open2(cx,cd,0);
  AVPacket*p=av_packet_alloc();AVFrame*f=av_frame_alloc();int got=0;
  while(got<1&&av_read_frame(fc,p)>=0){if(p->stream_index==si){avcodec_send_packet(cx,p);
    while(avcodec_receive_frame(cx,f)==0){const AVDRMFrameDescriptor*D=(void*)f->data[0];
      printf("frame %dx%d objs=%d layers=%d\n",f->width,f->height,D->nb_objects,D->nb_layers);
      for(int i=0;i<D->nb_objects;i++)printf("  obj%d fd=%d size=%zu mod=%llx\n",i,D->objects[i].fd,D->objects[i].size,(unsigned long long)D->objects[i].format_modifier);
      for(int l=0;l<D->nb_layers;l++){const AVDRMLayerDescriptor*L=&D->layers[l];printf("  layer%d fmt=%.4s planes=%d\n",l,(char*)&L->format,L->nb_planes);
        for(int i=0;i<L->nb_planes;i++)printf("    p%d obj=%d off=%td pitch=%td\n",i,L->planes[i].object_index,L->planes[i].offset,L->planes[i].pitch);}
      try_import(D,f->width,f->height,0,3);
      { AVDRMFrameDescriptor X=*D; X.layers[0].format=DRM_FORMAT_R8; X.layers[0].nb_planes=1; printf(" R8 Y-only:"); try_import(&X,f->width,f->height,0,1);
        X.layers[0].format=DRM_FORMAT_GR88; X.layers[0].planes[0]=D->layers[0].planes[1]; printf(" GR88 UV-only:"); try_import(&X,f->width/2,f->height/2,0,1); }
      got++; av_frame_unref(f);}}av_packet_unref(p);}
  av_frame_free(&f); av_packet_free(&p); avcodec_free_context(&cx); avformat_close_input(&fc);
  return 0;}
