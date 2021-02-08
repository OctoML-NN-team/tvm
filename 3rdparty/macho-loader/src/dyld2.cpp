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

extern "C" int vm_alloc(vm_address_t* addr, vm_size_t size, uint32_t flags)
{
    return ::vm_allocate(mach_task_self(), addr, size, flags);
}

extern "C" void* xmmap(void* addr, size_t len, int prot, int flags, int fd, off_t offset)
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


static const struct dyld_func dyld_funcs[] = {
        // TODO: fill it
//        {"__dyld_register_func_for_add_image",				(void*)_dyld_register_func_for_add_image },
//        {"__dyld_register_func_for_remove_image",			(void*)_dyld_register_func_for_remove_image },
//        {"__dyld_dladdr",									(void*)dladdr },
//        {"__dyld_dlclose",									(void*)dlclose },
//        {"__dyld_dlerror",									(void*)dlerror },
//        {"__dyld_dlopen_internal",							(void*)dlopen_internal },
//        {"__dyld_dlsym_internal",							(void*)dlsym_internal },
//        {"__dyld_dlopen_preflight_internal",				(void*)dlopen_preflight_internal },
//        {"__dyld_dlopen",									(void*)dlopen_compat },
//        {"__dyld_dlsym",									(void*)dlsym_compat },
//        {"__dyld_dlopen_preflight",							(void*)dlopen_preflight_compat },
//        {"__dyld_image_count",								(void*)_dyld_image_count },
//        {"__dyld_get_image_header",							(void*)_dyld_get_image_header },
//        {"__dyld_get_image_vmaddr_slide",					(void*)_dyld_get_image_vmaddr_slide },
//        {"__dyld_get_image_name",							(void*)_dyld_get_image_name },
//        {"__dyld_get_image_slide",							(void*)_dyld_get_image_slide },
//        {"__dyld_get_prog_image_header",					(void*)_dyld_get_prog_image_header },
//        {"__dyld__NSGetExecutablePath",						(void*)_NSGetExecutablePath },
//
//        // SPIs
//        {"__dyld_register_thread_helpers",					(void*)registerThreadHelpers },
//        {"__dyld_fork_child",								(void*)_dyld_fork_child },
//        {"__dyld_make_delayed_module_initializer_calls",	(void*)_dyld_make_delayed_module_initializer_calls },
//        {"__dyld_get_all_image_infos",						(void*)_dyld_get_all_image_infos },
//#if SUPPORT_ZERO_COST_EXCEPTIONS
//        {"__dyld_find_unwind_sections",						(void*)client_dyld_find_unwind_sections },
//#endif
//#if __i386__ || __x86_64__ || __arm__ || __arm64__
//        {"__dyld_fast_stub_entry",							(void*)dyld::fastBindLazySymbol },
//#endif
//        {"__dyld_image_path_containing_address",			(void*)dyld_image_path_containing_address },
//        {"__dyld_shared_cache_some_image_overridden",		(void*)dyld_shared_cache_some_image_overridden },
//        {"__dyld_process_is_restricted",					(void*)dyld::processIsRestricted },
//        {"__dyld_dynamic_interpose",						(void*)dyld_dynamic_interpose },
//        {"__dyld_shared_cache_file_path",					(void*)dyld::getStandardSharedCacheFilePath },
//        {"__dyld_has_inserted_or_interposing_libraries",	(void*)dyld::hasInsertedOrInterposingLibraries },
//        {"__dyld_get_image_header_containing_address",		(void*)dyld_image_header_containing_address },
//        {"__dyld_is_memory_immutable",						(void*)_dyld_is_memory_immutable },
//        {"__dyld_objc_notify_register",						(void*)_dyld_objc_notify_register },
//        {"__dyld_get_shared_cache_uuid",					(void*)_dyld_get_shared_cache_uuid },
//        {"__dyld_get_shared_cache_range",					(void*)_dyld_get_shared_cache_range },
//        {"__dyld_images_for_addresses",						(void*)_dyld_images_for_addresses },
//        {"__dyld_register_for_image_loads",					(void*)_dyld_register_for_image_loads },
//        {"__dyld_register_for_bulk_image_loads",			(void*)_dyld_register_for_bulk_image_loads },
//        {"__dyld_register_driverkit_main",					(void*)_dyld_register_driverkit_main },
//        {"__dyld_halt",										(void*)dyld::halt },
//
//#if DEPRECATED_APIS_SUPPORTED
//        #pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wdeprecated-declarations"
//    {"__dyld_lookup_and_bind",						(void*)client_dyld_lookup_and_bind },
//    {"__dyld_lookup_and_bind_with_hint",			(void*)_dyld_lookup_and_bind_with_hint },
//    {"__dyld_lookup_and_bind_fully",				(void*)_dyld_lookup_and_bind_fully },
//    {"__dyld_install_handlers",						(void*)_dyld_install_handlers },
//    {"__dyld_link_edit_error",						(void*)NSLinkEditError },
//    {"__dyld_unlink_module",						(void*)NSUnLinkModule },
//    {"__dyld_bind_fully_image_containing_address",  (void*)_dyld_bind_fully_image_containing_address },
//    {"__dyld_image_containing_address",				(void*)_dyld_image_containing_address },
//    {"__dyld_register_binding_handler",				(void*)_dyld_register_binding_handler },
//    {"__dyld_NSNameOfSymbol",						(void*)NSNameOfSymbol },
//    {"__dyld_NSAddressOfSymbol",					(void*)NSAddressOfSymbol },
//    {"__dyld_NSModuleForSymbol",					(void*)NSModuleForSymbol },
//    {"__dyld_NSLookupAndBindSymbol",				(void*)NSLookupAndBindSymbol },
//    {"__dyld_NSLookupAndBindSymbolWithHint",		(void*)NSLookupAndBindSymbolWithHint },
//    {"__dyld_NSLookupSymbolInModule",				(void*)NSLookupSymbolInModule},
//    {"__dyld_NSLookupSymbolInImage",				(void*)NSLookupSymbolInImage},
//    {"__dyld_NSMakePrivateModulePublic",			(void*)NSMakePrivateModulePublic},
//    {"__dyld_NSIsSymbolNameDefined",				(void*)client_NSIsSymbolNameDefined},
//    {"__dyld_NSIsSymbolNameDefinedWithHint",		(void*)NSIsSymbolNameDefinedWithHint },
//    {"__dyld_NSIsSymbolNameDefinedInImage",			(void*)NSIsSymbolNameDefinedInImage},
//    {"__dyld_NSNameOfModule",						(void*)NSNameOfModule },
//    {"__dyld_NSLibraryNameForModule",				(void*)NSLibraryNameForModule },
//    {"__dyld_NSAddLibrary",							(void*)NSAddLibrary },
//    {"__dyld_NSAddLibraryWithSearching",			(void*)NSAddLibraryWithSearching },
//    {"__dyld_NSAddImage",							(void*)NSAddImage },
//    {"__dyld_launched_prebound",					(void*)_dyld_launched_prebound },
//    {"__dyld_all_twolevel_modules_prebound",		(void*)_dyld_all_twolevel_modules_prebound },
//    {"__dyld_call_module_initializers_for_dylib",   (void*)_dyld_call_module_initializers_for_dylib },
//    {"__dyld_NSCreateObjectFileImageFromFile",			(void*)NSCreateObjectFileImageFromFile },
//    {"__dyld_NSCreateObjectFileImageFromMemory",		(void*)NSCreateObjectFileImageFromMemory },
//    {"__dyld_NSDestroyObjectFileImage",					(void*)NSDestroyObjectFileImage },
//    {"__dyld_NSLinkModule",								(void*)NSLinkModule },
//    {"__dyld_NSSymbolDefinitionCountInObjectFileImage",	(void*)NSSymbolDefinitionCountInObjectFileImage },
//    {"__dyld_NSSymbolDefinitionNameInObjectFileImage",	(void*)NSSymbolDefinitionNameInObjectFileImage },
//    {"__dyld_NSIsSymbolDefinedInObjectFileImage",		(void*)NSIsSymbolDefinedInObjectFileImage },
//    {"__dyld_NSSymbolReferenceNameInObjectFileImage",	(void*)NSSymbolReferenceNameInObjectFileImage },
//    {"__dyld_NSSymbolReferenceCountInObjectFileImage",	(void*)NSSymbolReferenceCountInObjectFileImage },
//    {"__dyld_NSGetSectionDataInObjectFileImage",		(void*)NSGetSectionDataInObjectFileImage },
//#if OLD_LIBSYSTEM_SUPPORT
//    {"__dyld_link_module",							(void*)_dyld_link_module },
//#endif
//#pragma clang diagnostic pop
//#endif //DEPRECATED_APIS_SUPPORTED

        {NULL, 0}
};

extern "C" int _dyld_func_lookup(const char* name, void** address)
{
    for (const dyld_func* p = dyld_funcs; p->name != NULL; ++p) {
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

void            tvm_stub_notifySingle(dyld_image_states, const ImageLoader* image, ImageLoader::InitializerTimingList*) {}
void            tvm_stub_notifyBatch(dyld_image_states state, bool preflightOnly) {}
void            tvm_stub_setErrorStrings(unsigned errorCode, const char* errorClientOfDylibPath,
                                         const char* errorTargetDylibPath, const char* errorSymbol) {}



void tvm_make_default_context(ImageLoader::LinkContext &ctx) {
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

