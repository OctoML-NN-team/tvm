<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

## Custom Mach-O Image Loader

### Purpose of this module
This is a simplified version of dynamic linker inherited from original Apple
sources. The key difference is in switched off signature check for loaded binaries.
Some platform like iOS doesn't provide mechanic to load unsigned binary but need it
for proper TVM RPC communication. Using this version of linker functionality allow
to get around this limitation.

This library exposes next symbols:
 - tvm_dlopen
 - tvm_dlclose
 - tvm_dlsym
 - tvm_dlerror

Use it instead of original Posix version.

### Known limitations
- Load only by absolute path
- There is no recurrent dependencies loading (all required modules should be
  preloaded in process before)
- Works only on system with enabled JIT permissions. Ex: iOS under debugger.
- Only RTLD_LAZY moade are supported 

### Borrowed files
- ImageLoader.h
- ImageLoader.cpp
- ImageLoaderMachO.h
- ImageLoaderMachO.cpp
- ImageLoaderMachOCompressed.h
- ImageLoaderMachOCompressed.cpp

All TVM specific changes of original code are under conditional compilation with 
`#ifdef FOR_TVM` macros.

The file `dyld_stubs.cpp` contains some utils and other stub functions to make
this code compilable. Most of them has no implementation, just a compatible 
signature. 

### Link on original sources
https://opensource.apple.com/source/dyld/dyld-832.7.3
