// yuvhint: import a linear NV12 dma-buf into Mesa with each EGL YUV colour-space/range hint and print
// the RGB it samples. Run it with LD_PRELOAD=libegl_linear_shim.so EGL_SHIM_LIMITED_OUT=1 to check the
// shim relabel (the "709 narrow" row must then match "709 full"). Args: Y U V (default: BT.709 red).
// Build: gcc -O2 -I/usr/include/libdrm -o tests/bin/yuvhint tests/yuvhint.c -l:libgbm.so.1 -l:libEGL.so.1 -l:libGLESv2.so.2
#include <stdlib.h>
// yuvhint: import a linear NV12 dmabuf into Mesa with different EGL YUV hints and read back RGB.
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <drm_fourcc.h>
#include <stdint.h>
#include "egl_min.h"
struct gbm_device; struct gbm_bo;
struct gbm_device*gbm_create_device(int); struct gbm_bo*gbm_bo_create(struct gbm_device*,uint32_t,uint32_t,uint32_t,uint32_t);
void*gbm_bo_map(struct gbm_bo*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t*,void**); void gbm_bo_unmap(struct gbm_bo*,void*); int gbm_bo_get_fd(struct gbm_bo*);
#define GBM_FORMAT_R8 0x20203852
#define GBM_BO_USE_LINEAR (1<<4)
#define GBM_BO_USE_RENDERING (1<<2)
#define GBM_BO_TRANSFER_WRITE 2
typedef unsigned int GLuint,GLenum; typedef int GLint,GLsizei; typedef float GLfloat; typedef unsigned char GLboolean; typedef char GLchar;
#define GL_TEXTURE_2D 0x0DE1
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_FLOAT 0x1406
#define GL_TRIANGLE_STRIP 5
#define GL_TEXTURE_EXTERNAL_OES 0x8D65
void glGenTextures(GLsizei,GLuint*);void glBindTexture(GLenum,GLuint);void glTexImage2D(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*);
void glGenFramebuffers(GLsizei,GLuint*);void glBindFramebuffer(GLenum,GLuint);void glFramebufferTexture2D(GLenum,GLenum,GLenum,GLuint,GLint);
GLuint glCreateProgram(void);GLuint glCreateShader(GLenum);void glShaderSource(GLuint,GLsizei,const GLchar*const*,const GLint*);void glCompileShader(GLuint);
void glAttachShader(GLuint,GLuint);void glBindAttribLocation(GLuint,GLuint,const GLchar*);void glLinkProgram(GLuint);void glUseProgram(GLuint);
void glVertexAttribPointer(GLuint,GLint,GLenum,GLboolean,GLsizei,const void*);void glEnableVertexAttribArray(GLuint);void glViewport(GLint,GLint,GLsizei,GLsizei);
void glDrawArrays(GLenum,GLint,GLsizei);void glReadPixels(GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,void*);
typedef void(*PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)(GLenum,void*);
#ifndef EGL_OPENGL_ES_API
#define EGL_OPENGL_ES_API 0x30A0
#endif
#define EGL_CONTEXT_CLIENT_VERSION 0x3098
#define EGL_NO_CONFIG_KHR ((void*)0)
#define EGL_NO_SURFACE ((void*)0)
#define EGL_YUV_COLOR_SPACE_HINT_EXT 0x327B
#define EGL_SAMPLE_RANGE_HINT_EXT 0x327C
#define EGL_ITU_REC601_EXT 0x327F
#define EGL_ITU_REC709_EXT 0x3280
#define EGL_ITU_REC2020_EXT 0x3281
#define EGL_YUV_FULL_RANGE_EXT 0x3282
#define EGL_YUV_NARROW_RANGE_EXT 0x3283
unsigned int eglBindAPI(unsigned int); void*eglCreateContext(void*,void*,void*,const int*); unsigned int eglMakeCurrent(void*,void*,void*,void*);
#define W 64
#define H 64
static const char*vs="attribute vec2 p;varying vec2 t;void main(){t=p*0.5+0.5;gl_Position=vec4(p,0,1);}";
static const char*fs="#extension GL_OES_EGL_image_external:require\nprecision mediump float;uniform samplerExternalOES s;varying vec2 t;void main(){gl_FragColor=texture2D(s,t);}";
int main(int argc,char**argv){
  int Y=63,U=102,V=240; if(argc==4){Y=atoi(argv[1]);U=atoi(argv[2]);V=atoi(argv[3]);}
  int fd=open("/dev/dri/renderD128",O_RDWR); struct gbm_device*g=gbm_create_device(fd);
  struct gbm_bo*bo=gbm_bo_create(g,W,H*3/2,GBM_FORMAT_R8,GBM_BO_USE_LINEAR|GBM_BO_USE_RENDERING);
  if(!bo){fprintf(stderr,"gbm bo\n");return 1;}
  uint32_t st; void*md=NULL; uint8_t*m=gbm_bo_map(bo,0,0,W,H*3/2,GBM_BO_TRANSFER_WRITE,&st,&md);
  for(int y=0;y<H;y++) memset(m+y*st,Y,W);
  for(int y=0;y<H/2;y++) for(int x=0;x<W;x+=2){m[(H+y)*st+x]=U;m[(H+y)*st+x+1]=V;}
  gbm_bo_unmap(bo,md); int dfd=gbm_bo_get_fd(bo);
  PFNEGLGETPLATFORMDISPLAYEXTPROC gp=(void*)eglGetProcAddress("eglGetPlatformDisplayEXT"); EGLDisplay d=gp(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,NULL); eglInitialize(d,0,0); eglBindAPI(EGL_OPENGL_ES_API);
  EGLint ca[]={EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE}; EGLContext c=eglCreateContext(d,EGL_NO_CONFIG_KHR,EGL_NO_CONTEXT,ca);
  eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,c);
  PFNEGLCREATEIMAGEKHRPROC ci=(void*)eglGetProcAddress("eglCreateImageKHR");
  PFNGLEGLIMAGETARGETTEXTURE2DOESPROC tt=(void*)eglGetProcAddress("glEGLImageTargetTexture2DOES");
  GLuint rt,fb; glGenTextures(1,&rt); glBindTexture(GL_TEXTURE_2D,rt); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,0);
  glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,rt,0);
  GLuint p=glCreateProgram(),a=glCreateShader(GL_VERTEX_SHADER),b=glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(a,1,&vs,0);glCompileShader(a);glShaderSource(b,1,&fs,0);glCompileShader(b);glAttachShader(p,a);glAttachShader(p,b);glBindAttribLocation(p,0,"p");glLinkProgram(p);glUseProgram(p);
  float q[]={-1,-1,1,-1,-1,1,1,1}; glVertexAttribPointer(0,2,GL_FLOAT,0,0,q); glEnableVertexAttribArray(0); glViewport(0,0,W,H);
  struct{const char*n;EGLint cs,rg;}t[]={{"601 narrow",EGL_ITU_REC601_EXT,EGL_YUV_NARROW_RANGE_EXT},{"709 narrow",EGL_ITU_REC709_EXT,EGL_YUV_NARROW_RANGE_EXT},{"709 full",EGL_ITU_REC709_EXT,EGL_YUV_FULL_RANGE_EXT},{"2020 narrow",EGL_ITU_REC2020_EXT,EGL_YUV_NARROW_RANGE_EXT},{"(no hint)",0,0}};
  printf("NV12 Y=%d U=%d V=%d\n",Y,U,V);
  for(int i=0;i<5;i++){
    EGLint at[40]={EGL_WIDTH,W,EGL_HEIGHT,H,EGL_LINUX_DRM_FOURCC_EXT,DRM_FORMAT_NV12,
      EGL_DMA_BUF_PLANE0_FD_EXT,dfd,EGL_DMA_BUF_PLANE0_OFFSET_EXT,0,EGL_DMA_BUF_PLANE0_PITCH_EXT,(EGLint)st,
      EGL_DMA_BUF_PLANE1_FD_EXT,dfd,EGL_DMA_BUF_PLANE1_OFFSET_EXT,(EGLint)(st*H),EGL_DMA_BUF_PLANE1_PITCH_EXT,(EGLint)st,
      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,0,EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,0,EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,0,EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,0};
    int k=26; if(t[i].cs){at[k++]=EGL_YUV_COLOR_SPACE_HINT_EXT;at[k++]=t[i].cs;at[k++]=EGL_SAMPLE_RANGE_HINT_EXT;at[k++]=t[i].rg;} at[k]=EGL_NONE;
    EGLImageKHR img=ci(d,EGL_NO_CONTEXT,EGL_LINUX_DMA_BUF_EXT,0,at); if(!img){printf("%-12s import fail 0x%x\n",t[i].n,eglGetError());continue;}
    GLuint tx; glGenTextures(1,&tx); glBindTexture(GL_TEXTURE_EXTERNAL_OES,tx); tt(GL_TEXTURE_EXTERNAL_OES,img);
    glDrawArrays(GL_TRIANGLE_STRIP,0,4); uint8_t px[4]; glReadPixels(W/2,H/2,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);
    printf("%-12s -> R=%3d G=%3d B=%3d\n",t[i].n,px[0],px[1],px[2]);}
  return 0;}
