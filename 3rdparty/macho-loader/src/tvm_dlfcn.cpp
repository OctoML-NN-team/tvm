
#include <tvm_dlfcn.h>
#include "ImageLoaderMachO.h"
#include "unsafe_loader.h"

#include <fstream>
#include <dlfcn.h>

using namespace tvm_exp;

ImageLoader::LinkContext tvm_linkContext = {};

extern "C" void tvm_make_default_context(ImageLoader::LinkContext &ctx);

extern "C" int tvm_dlclose(void * __handle) {
  ImageLoader* image = reinterpret_cast<ImageLoader*>(__handle);
  ImageLoader::deleteImage(image);
  return 0;
}

extern "C" char * tvm_dlerror(void) {
  return "";
}

extern "C" void* tvm_dlopen(const char * __path, int __mode) {
  tvm_make_default_context(tvm_linkContext);
  ImageLoader::LinkContext &linkContext = tvm_linkContext;
  const char stub_name[] = "no_name"; // TODO: extract form path

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

  auto mh = reinterpret_cast<const macho_header*>(buff.data());
  auto image = ImageLoaderMachO::instantiateFromMemory(stub_name, mh, fsize, linkContext);

  bool forceLazysBound = true;
  bool preflightOnly = false;
  bool neverUnload = false;

  std::vector<const char*> rpathsFromCallerImage;
  ImageLoader::RPathChain loaderRPaths(NULL, &rpathsFromCallerImage);

  image->link(linkContext, forceLazysBound, preflightOnly, neverUnload, loaderRPaths, __path);
  return image;
}

extern "C" void* tvm_dlsym(void * __handle, const char * __symbol) {
  std::string underscoredName = "_" + std::string(__symbol);
  const ImageLoader* image = reinterpret_cast<ImageLoader*>(__handle);

  auto sym = image->findExportedSymbol(underscoredName.c_str(), true, &image);
  if ( sym != NULL ) {
    auto addr = image->getExportedSymbolAddress(sym,
            tvm_linkContext, nullptr, false, underscoredName.c_str());
    return reinterpret_cast<void*>(addr);
  }
  return nullptr;
}

