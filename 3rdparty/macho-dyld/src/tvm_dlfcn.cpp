
#include <tvm_dlfcn.h>

#include "dyld/ImageLoaderMachO.h"
#include <fstream>

using namespace tvm_exp;

ImageLoader::LinkContext tvm_linkContext = {};

extern "C" void tvm_make_default_context(ImageLoader::LinkContext &ctx);

thread_local char* _err_buf = nullptr;
thread_local size_t _err_buf_size = 0;

void tvm_clean_error() {
  if (_err_buf)
    _err_buf[0] = 0;
}

void tvm_set_dlerror(const std::string &msg) {
  if (msg.size() >= _err_buf_size) {
    if (_err_buf)
      free(_err_buf);

    _err_buf = static_cast<char*>(malloc(msg.size() + 1));
    _err_buf_size = msg.size() + 1;
  }
  strcpy(_err_buf, msg.c_str());
}

extern "C" char* tvm_dlerror(void) {
  if (_err_buf && _err_buf[0] == 0)
    return nullptr;

  return _err_buf;
}

static bool is_absolute_path(const char * path) {
  return path && path[0] == '/';
}

std::string base_name(const std::string &path) {
  return path.substr(path.find_last_of("/\\") + 1);
}

void* with_error(std::string str) {
  tvm_set_dlerror(str);
  return nullptr;
}

extern "C" void* tvm_dlopen(const char * __path, int __mode) {
  tvm_clean_error();
  if (!is_absolute_path(__path))
    return with_error("You are using custom TVM mach-o dyld. Only absolute path is supported. "
                      "Please specify full path to binary.");

  std::fstream lib_f(__path, std::ios::in | std::ios::binary);
  if (!lib_f.is_open())
    return with_error("File is not found.");

  try {
    std::streampos fsize = lib_f.tellg();
    lib_f.seekg(0, std::ios::end);
    fsize = lib_f.tellg() - fsize;
    lib_f.seekg(0, std::ios::beg);

    std::vector<char> buff(fsize);
    lib_f.read(buff.data(), fsize);
    lib_f.close();

    tvm_make_default_context(tvm_linkContext);
    ImageLoader::LinkContext& linkContext = tvm_linkContext;
    std::string file_name = base_name(__path);

    auto mh = reinterpret_cast<const macho_header*>(buff.data());
    auto image = ImageLoaderMachO::instantiateFromMemory(file_name.c_str(), mh, fsize, linkContext);

    bool forceLazysBound = true;
    bool preflightOnly = false;
    bool neverUnload = false;

    std::vector<const char*> rpathsFromCallerImage;
    ImageLoader::RPathChain loaderRPaths(NULL, &rpathsFromCallerImage);

    image->link(linkContext, forceLazysBound, preflightOnly, neverUnload, loaderRPaths, __path);
    return image;
  } catch (std::logic_error e) {
    return with_error(std::string("Error happens during dlopen execution. ") + e.what());
  } catch (char * msg) {
    return with_error(std::string("Error happens during dlopen execution. ") + msg);
  }
}

extern "C" void* tvm_dlsym(void * __handle, const char * __symbol) {
  std::string underscoredName = "_" + std::string(__symbol);
  const ImageLoader* image = reinterpret_cast<ImageLoader*>(__handle);

  tvm_clean_error();
  try {
    auto sym = image->findExportedSymbol(underscoredName.c_str(), true, &image);
    if (sym != NULL) {
      auto addr = image->getExportedSymbolAddress(sym, tvm_linkContext, nullptr, false,
                                                  underscoredName.c_str());
      return reinterpret_cast<void*>(addr);
    }
    return with_error("No symbol found.");
  } catch (std::logic_error e) {
    return with_error(std::string("Error happens during dlsym execution. ") + e.what());
  } catch (char * msg) {
    return with_error(std::string("Error happens during dlsym execution. ") + msg);
  }
}

extern "C" int tvm_dlclose(void * __handle) {
  ImageLoader* image = reinterpret_cast<ImageLoader*>(__handle);
  ImageLoader::deleteImage(image);
  return 0;
}
