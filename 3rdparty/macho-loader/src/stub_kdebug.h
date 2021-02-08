//
// Created by Alexander Peskov on 04.02.2021.
//

#ifndef TVM_LOADER_STUB_KDEBUG_H
#define TVM_LOADER_STUB_KDEBUG_H

#define KDBG_CODE(Class, SubClass, code) (((Class & 0xff) << 24) | ((SubClass & 0xff) << 16) | ((code & 0x3fff)  << 2))

#define DBG_DYLD            31

#endif //TVM_LOADER_STUB_KDEBUG_H
