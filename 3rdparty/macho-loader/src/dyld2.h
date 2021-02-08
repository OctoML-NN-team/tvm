//
// Created by Alexander Peskov on 04.02.2021.
//

#ifndef TVM_LOADER_DYLD2_H
#define TVM_LOADER_DYLD2_H

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdexcept>

#define ImageLoaderMachOCompressed ImageLoaderMachOCompressed_
#define ImageLoaderMachO ImageLoaderMachO_
#define ImageLoader ImageLoader_

struct dyld_interpose_tuple {
    const void* replacement;
    const void* replacee;
};

// header of the LC_DYLD_CHAINED_FIXUPS payload
struct dyld_chained_fixups_header
{
    uint32_t    fixups_version;    // 0
    uint32_t    starts_offset;     // offset of dyld_chained_starts_in_image in chain_data
    uint32_t    imports_offset;    // offset of imports table in chain_data
    uint32_t    symbols_offset;    // offset of symbol strings in chain_data
    uint32_t    imports_count;     // number of imported symbol names
    uint32_t    imports_format;    // DYLD_CHAINED_IMPORT*
    uint32_t    symbols_format;    // 0 => uncompressed, 1 => zlib compressed
};

//
// When dyld must terminate a process because of a required dependent dylib
// could not be loaded or a symbol is missing, dyld calls abort_with_reason()
// using one of the following error codes.
//
#define DYLD_EXIT_REASON_DYLIB_MISSING          1
#define DYLD_EXIT_REASON_DYLIB_WRONG_ARCH       2
#define DYLD_EXIT_REASON_DYLIB_WRONG_VERSION    3
#define DYLD_EXIT_REASON_SYMBOL_MISSING         4
#define DYLD_EXIT_REASON_CODE_SIGNATURE         5
#define DYLD_EXIT_REASON_FILE_SYSTEM_SANDBOX    6
#define DYLD_EXIT_REASON_MALFORMED_MACHO        7
#define DYLD_EXIT_REASON_OTHER                  9


namespace dyld {

    struct dyld_all_image_infos{
        bool libSystemInitialized = true;
    };
    extern struct dyld_all_image_infos*	gProcessInfo;

    struct LibSystemHelpers {};
    extern const struct LibSystemHelpers* gLibSystemHelpers;
}

extern "C" void* tvm_find_exterm_sym(const char * sym_mane);
    
#endif //TVM_LOADER_DYLD2_H
