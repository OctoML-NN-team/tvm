//
// Created by Alexander Peskov on 11.02.2021.
//

#ifndef TVM_UNSAFE_LOADER_H
#define TVM_UNSAFE_LOADER_H

#include <vector>
#include <mach-o/loader.h>

namespace tvm {

struct LinkContext {
  bool strictMachORequired = true;
  bool verboseMapping = false;
  bool verboseBind = false;
};

using macho_header = mach_header_64;
using macho_segment_command = segment_command_64;

class unsafe_loader {
 public:
  static unsafe_loader* loadFromMemory(std::vector<char> buff);
  static void deleteImage(unsafe_loader* image);
  void* getExportSymbol(const char* sym) const;

 private:

  unsafe_loader(const macho_header* mh, const char* path, unsigned int segCount,
                uint32_t segOffsets[], unsigned int libCount);

  ~unsafe_loader();

  void destroy();

  static void sniffLoadCommands(const macho_header* mh, const char* path, bool inCache,
                         bool* compressed, unsigned int* segCount,
                         unsigned int* libCount, const LinkContext& context,
                         const linkedit_data_command** codeSigCmd,
                         const encryption_info_command** encryptCmd);

  static bool needsAddedLibSystemDepency(unsigned int libCount, const macho_header* mh);

  static unsafe_loader* instantiateFromMemory(const char* moduleName, const macho_header* mh, uint64_t len,
                                              unsigned int segCount, unsigned int libCount, const LinkContext& context);

  static unsafe_loader* instantiateStart(const macho_header* mh, const char* path,
                                         unsigned int segCount, unsigned int libCount);


  void parseLoadCmds(const LinkContext& context);
  void mapSegments(const void* memoryImage, uint64_t imageLen, const LinkContext& context);
  void UnmapSegments();

  intptr_t assignSegmentAddresses(const LinkContext& context, size_t extraAllocationSize);



  uint32_t* segmentCommandOffsets() const;
  const macho_segment_command* segLoadCommand(unsigned int segIndex) const;

  const uint8_t* trieWalk(const uint8_t* start, const uint8_t* end, const char* s) const;

  bool segHasPreferredLoadAddress(unsigned int segIndex) const;
  const char* segName(unsigned int segIndex) const;
  uintptr_t segFileOffset(unsigned int segIndex) const;
  uintptr_t segPreferredLoadAddress(unsigned int segIndex) const;
  uintptr_t segFileSize(unsigned int segIndex) const;
  uintptr_t segSize(unsigned int segIndex) const;
  uintptr_t segActualLoadAddress(unsigned int segIndex) const;
  uintptr_t segActualEndAddress(unsigned int segIndex) const;


  struct Symbol {
  };

  const unsafe_loader::Symbol* findShallowExportedSymbol(const char* symbol) const;
  uintptr_t exportedSymbolAddress(const LinkContext& context, const Symbol* symbol, bool runResolver) const;

  std::vector<char> buff;

  const struct dyld_info_command*		fDyldInfo;
  const struct linkedit_data_command*		fChainedFixups;
  const struct linkedit_data_command*		fExportsTrie;

  uint32_t fSegmentsCount;
  uint32_t fIsSplitSeg;
  uint32_t fInSharedCache;
  uint32_t fHasSubLibraries;
  uint32_t fHasSubUmbrella;
  uint32_t fInUmbrella;
  uint32_t fHasDOFSections;
  uint32_t fHasDashInit;
  uint32_t fHasInitializers;
  uint32_t fHasTerminators;
  uint32_t fEHFrameSectionOffset;
  uint32_t fUnwindInfoSectionOffset;
  uint32_t fDylibIDOffset;

  uintptr_t fSlide;
  const uint8_t* fMachOData;
  const uint8_t* fLinkEditBase;
};
}

#endif  // TVM_UNSAFE_LOADER_H
