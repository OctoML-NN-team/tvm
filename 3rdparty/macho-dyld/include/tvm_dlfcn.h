//
// Created by Alexander Peskov on 04.02.2021.
//

#ifndef TVM_TVM_DLFCN_H
#define TVM_TVM_DLFCN_H

#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int tvm_dlclose(void * __handle);
extern char * tvm_dlerror(void);
extern void * tvm_dlopen(const char * __path, int __mode);
extern void * tvm_dlsym(void * __handle, const char * __symbol);

#ifdef __cplusplus
}
#endif

#endif //TVM_TVM_DLFCN_H
