
#include <tvm_dlfcn.h>
#include "ImageLoaderMachO.h"

#include <fstream>

ImageLoader::LinkContext tvm_linkContext = {};

extern void tvm_make_default_context(ImageLoader::LinkContext &ctx);


extern "C" int tvm_dlclose(void * __handle) {
  ImageLoader* image = reinterpret_cast<ImageLoader*>(__handle);
  ImageLoader::deleteImage(image);
  return 0;
}

extern "C" char * tvm_dlerror(void) {
  return "";
}

extern "C" void * tvm_dlopen(const char * __path, int __mode) {
  tvm_make_default_context(tvm_linkContext);

  ImageLoader* image = nullptr;
  ImageLoader::LinkContext &linkContext = tvm_linkContext;
  const char stub_name[] = "libstub.dylib"; // TODO: extract form path

  std::fstream lib_f(__path, std::ios::in | std::ios::binary);
  if (!lib_f.is_open())
    return nullptr;
  std::streampos fsize = lib_f.tellg();
  lib_f.seekg( 0, std::ios::end );
  fsize = lib_f.tellg() - fsize;
  lib_f.seekg( 0, std::ios::beg );

  std::vector<char> buff(fsize);
  lib_f.read(buff.data(), fsize);
  lib_f.close();

  macho_header* mh = reinterpret_cast<macho_header*>(buff.data());

  image = ImageLoaderMachO::instantiateFromMemory(stub_name, mh, fsize, linkContext);

  bool forceLazysBound = true;
  bool preflightOnly = false;
  bool neverUnload = false;

  std::vector<const char*> rpathsFromCallerImage;
  ImageLoader::RPathChain loaderRPaths(NULL, &rpathsFromCallerImage);

  image->link(linkContext, forceLazysBound, preflightOnly, neverUnload, loaderRPaths, __path);
  return image;
}

extern "C" void * tvm_dlsym(void * __handle, const char * __symbol) {
  const ImageLoader* image = reinterpret_cast<ImageLoader*>(__handle);
  const ImageLoader::Symbol* sym;
  void* result;

  ImageLoader* callerImage = nullptr;
  ImageLoader::LinkContext linkContext = tvm_linkContext;
  std::string underscoredName = "_" + std::string(__symbol);

  sym = image->findExportedSymbol(underscoredName.c_str(), true, &image);
  if ( sym != NULL ) {
    result = (void*)image->getExportedSymbolAddress(sym,
                    linkContext, callerImage, false, underscoredName.c_str());
    return result;
  }
  return nullptr;
}

