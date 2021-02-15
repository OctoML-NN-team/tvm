//
// Created by Alexander Peskov on 04.02.2021.
//

#include "dyld2.h"
#include <mach/vm_types.h>
#include <mach/vm_map.h>
#include <mach/mach_init.h>

#include <sys/mman.h>

#include "ImageLoader.h"

#include <dlfcn.h>
#include <string>

namespace tvm_exp {

namespace dyld {

    struct dyld_all_image_infos dyld_all_image_infos;
    struct dyld_all_image_infos *gProcessInfo = &dyld_all_image_infos;

    const struct LibSystemHelpers *gLibSystemHelpers = NULL;
    void throwf(const char* format, ...)
    {
        char buf[4096*10];
        va_list    list;
        va_start(list, format);
        sprintf(buf, format, list);
        throw std::runtime_error(buf);
    }
    void log(const char* format, ...)
    {
        va_list	list;
        va_start(list, format);
        printf(format, list);
        va_end(list);
    }

    void warn(const char* format, ...)
    {
        va_list	list;
        va_start(list, format);
        printf(format, list);
        va_end(list);
    }

    const char* mkstringf(const char* format, ...)
    {
        // TODO: TBD
        return "mkstringf, cannot create string";
    }

}

extern "C" int vm_alloc__(vm_address_t* addr, vm_size_t size, uint32_t flags)
{
    return ::vm_allocate(mach_task_self(), addr, size, flags);
}

extern "C" void* xmmap__(void* addr, size_t len, int prot, int flags, int fd, off_t offset)
{
    return ::mmap(addr, len, prot, flags, fd, offset);
}

struct dyld_func {
    const char* name;
    void*		implementation;
};

static void unimplemented()
{
//    dyld::halt("unimplemented dyld function\n");
    printf("unimplemented dyld function\n");
}


static const struct dyld_func dyld_funcs__bla[] = {
    // Empty Stub registry
    {NULL, 0}
};

extern "C" int _dyld_func_lookup__(const char* name, void** address)
{
    for (const dyld_func* p = dyld_funcs__bla; p->name != NULL; ++p) {
        if ( strcmp(p->name, name) == 0 ) {
            if( p->implementation == unimplemented )
                dyld::log("unimplemented dyld function: %s\n", p->name);
            *address = p->implementation;
            return true;
        }
    }
    *address = 0;
    return false;
}

void tvm_stub_notifySingle(dyld_image_states, const ImageLoader* image, ImageLoader::InitializerTimingList*) {}
void tvm_stub_notifyBatch(dyld_image_states state, bool preflightOnly) {}
void tvm_stub_setErrorStrings(unsigned errorCode, const char* errorClientOfDylibPath,
                                         const char* errorTargetDylibPath, const char* errorSymbol) {}

extern "C" void tvm_make_default_context(ImageLoader::LinkContext &ctx) {
    ctx.bindFlat = true;
    ctx.prebindUsage = ImageLoader::kUseNoPrebinding;
    ctx.notifyBatch = tvm_stub_notifyBatch;
    ctx.notifySingle = tvm_stub_notifySingle;
    ctx.setErrorStrings = tvm_stub_setErrorStrings;
}

extern "C" void* tvm_find_exterm_sym(const char * sym_name) {
    if (std::string(sym_name) == "dyld_stub_binder")
        return  (void*)tvm_find_exterm_sym;
    if (sym_name[0] == '_')
        sym_name ++;
    void * handle = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
    return dlsym(handle, sym_name);
}

}
