
#include "unsafe_loader.h"

#include <mach-o/loader.h>
#include <mach-o/nlist.h>

namespace tvm {

using macho_section = section_64;
using macho_nlist = nlist_64;
#define LC_SEGMENT_COMMAND       LC_SEGMENT_64
#define LC_ROUTINES_COMMAND      LC_ROUTINES_64
#define LC_SEGMENT_COMMAND_WRONG LC_SEGMENT

#define dyld_page_trunc(__addr)     (__addr & (-16384))
#define dyld_page_round(__addr)     ((__addr + 16383) & (-16384))
#define dyld_page_size              16384

void throwf_(const char* format, ...)
{
  char buf[4096*10];
  va_list    list;
  va_start(list, format);
  sprintf(buf, format, list);
  throw std::runtime_error(buf);
}

void log_(const char* format, ...)
{
  va_list	list;
  va_start(list, format);
  printf(format, list);
  va_end(list);
}

void unsafe_loader::deleteImage(unsafe_loader* image)
{
  delete image;
}

unsafe_loader::~unsafe_loader()
{
  // don't do clean up in ~ImageLoaderMachO() because virtual call to segmentCommandOffsets() won't work
  destroy();
}

// don't do this work in destructor because we need object to be full subclass
// for UnmapSegments() to work
void unsafe_loader::destroy()
{
  // unmap image when done
  UnmapSegments();
}

void unsafe_loader::UnmapSegments()
{
  free((void*)fSlide);
  // usually unmap image when done
//  if ( ! this->leaveMapped() && (this->() >= dyld_image_state_mapped) ) {
//    // unmap TEXT segment last because it contains load command being inspected
//    unsigned int textSegmentIndex = 0;
//    for(unsigned int i=0; i < fSegmentsCount; ++i) {
//      //dyld::log("unmap %s at 0x%08lX\n", seg->getName(), seg->getActualLoadAddress(this));
//      if ( (segFileOffset(i) == 0) && (segFileSize(i) != 0) ) {
//        textSegmentIndex = i;
//      }
//      else {
//        // update stats
//        --ImageLoader::fgTotalSegmentsMapped;
//        ImageLoader::fgTotalBytesMapped -= segSize(i);
//        munmap((void*)segActualLoadAddress(i), segSize(i));
//      }
//    }
//    // now unmap TEXT
//    --ImageLoader::fgTotalSegmentsMapped;
//    ImageLoader::fgTotalBytesMapped -= segSize(textSegmentIndex);
//    munmap((void*)segActualLoadAddress(textSegmentIndex), segSize(textSegmentIndex));
//  }
}

unsafe_loader* unsafe_loader::loadFromMemory(std::vector<char> buff_) {
  auto *mh = reinterpret_cast<macho_header*>(buff_.data());
  uint64_t len = buff_.size();
  auto ctx = tvm::LinkContext {};


  const char* moduleName = "no_name";
  bool compressed;
  unsigned int segCount;
  unsigned int libCount;
  const linkedit_data_command* sigcmd;
  const encryption_info_command* encryptCmd;
  sniffLoadCommands(mh, moduleName, false, &compressed, &segCount, &libCount, ctx, &sigcmd, &encryptCmd);
  // instantiate concrete class based on content of load commands
  if ( compressed )
    return unsafe_loader::instantiateFromMemory(moduleName, mh, len, segCount, libCount, ctx);
  else
    tvm::throwf_("Only compressed module version is supported");
  return nullptr;
}


// determine if this mach-o file has classic or compressed LINKEDIT and number of segments it has
void unsafe_loader::sniffLoadCommands(const macho_header* mh, const char* path, bool inCache,
                                      bool* compressed, unsigned int* segCount,
                                      unsigned int* libCount, const LinkContext& context,
                                      const linkedit_data_command** codeSigCmd,
                                      const encryption_info_command** encryptCmd) {
  *compressed = false;
  *segCount = 0;
  *libCount = 0;
  *codeSigCmd = NULL;
  *encryptCmd = NULL;

  const uint32_t cmd_count = mh->ncmds;
  const uint32_t sizeofcmds = mh->sizeofcmds;
  if (cmd_count > (sizeofcmds / sizeof(load_command)))
    tvm::throwf_("malformed mach-o: ncmds (%u) too large to fit in sizeofcmds (%u)", cmd_count,
                 sizeofcmds);
  const struct load_command* const startCmds =
      (struct load_command*)(((uint8_t*)mh) + sizeof(macho_header));
  const struct load_command* const endCmds =
      (struct load_command*)(((uint8_t*)mh) + sizeof(macho_header) + sizeofcmds);
  const struct load_command* cmd = startCmds;
  bool foundLoadCommandSegment = false;
  const macho_segment_command* linkeditSegCmd = NULL;
  const macho_segment_command* startOfFileSegCmd = NULL;
  const dyld_info_command* dyldInfoCmd = NULL;
  const linkedit_data_command* chainedFixupsCmd = NULL;
  const linkedit_data_command* exportsTrieCmd = NULL;
  const symtab_command* symTabCmd = NULL;
  const dysymtab_command* dynSymbTabCmd = NULL;
  for (uint32_t i = 0; i < cmd_count; ++i) {
    uint32_t cmdLength = cmd->cmdsize;
    const macho_segment_command* segCmd;
    const dylib_command* dylibCmd;
    if (cmdLength < 8) {
      tvm::throwf_("malformed mach-o image: load command #%d length (%u) too small in %s", i,
                   cmdLength, path);
    }
    const struct load_command* const nextCmd =
        (const struct load_command*)(((char*)cmd) + cmdLength);
    if ((nextCmd > endCmds) || (nextCmd < cmd)) {
      tvm::throwf_(
          "malformed mach-o image: load command #%d length (%u) would exceed sizeofcmds (%u) in %s",
          i, cmdLength, mh->sizeofcmds, path);
    }
    switch (cmd->cmd) {
      case LC_DYLD_INFO:
      case LC_DYLD_INFO_ONLY:
        if (cmd->cmdsize != sizeof(dyld_info_command))
          throw "malformed mach-o image: LC_DYLD_INFO size wrong";
        dyldInfoCmd = (struct dyld_info_command*)cmd;
        *compressed = true;
        break;
      case LC_DYLD_CHAINED_FIXUPS:
        if (cmd->cmdsize != sizeof(linkedit_data_command))
          throw "malformed mach-o image: LC_DYLD_CHAINED_FIXUPS size wrong";
        chainedFixupsCmd = (struct linkedit_data_command*)cmd;
        *compressed = true;
        break;
      case LC_DYLD_EXPORTS_TRIE:
        if (cmd->cmdsize != sizeof(linkedit_data_command))
          throw "malformed mach-o image: LC_DYLD_EXPORTS_TRIE size wrong";
        exportsTrieCmd = (struct linkedit_data_command*)cmd;
        break;
      case LC_SEGMENT_COMMAND:
        segCmd = (macho_segment_command*)cmd;
        // <rdar://problem/19986776> dyld should support non-allocatable __LLVM segment
        if ((segCmd->filesize > segCmd->vmsize) &&
            ((segCmd->vmsize != 0) || ((segCmd->flags & SG_NORELOC) == 0)))
          tvm::throwf_(
              "malformed mach-o image: segment load command %s filesize (0x%0lX) is larger than vmsize (0x%0lX)",
              segCmd->segname, (long)segCmd->filesize, (long)segCmd->vmsize);
        if (cmd->cmdsize < sizeof(macho_segment_command))
          throw "malformed mach-o image: LC_SEGMENT size too small";
        if (cmd->cmdsize !=
            (sizeof(macho_segment_command) + segCmd->nsects * sizeof(macho_section)))
          throw "malformed mach-o image: LC_SEGMENT size wrong for number of sections";
        // ignore zero-sized segments
        if (segCmd->vmsize != 0) *segCount += 1;
        if (strcmp(segCmd->segname, "__LINKEDIT") == 0) {
          if (segCmd->fileoff == 0)
            throw "malformed mach-o image: __LINKEDIT has fileoff==0 which overlaps mach_header";
          if (linkeditSegCmd != NULL) throw "malformed mach-o image: multiple __LINKEDIT segments";
          linkeditSegCmd = segCmd;
        } else {
          if (segCmd->initprot & 0xFFFFFFF8)
            tvm::throwf_(
                "malformed mach-o image: %s segment has invalid permission bits (0x%X) in initprot",
                segCmd->segname, segCmd->initprot);
          if (segCmd->maxprot & 0xFFFFFFF8)
            tvm::throwf_(
                "malformed mach-o image: %s segment has invalid permission bits (0x%X) in maxprot",
                segCmd->segname, segCmd->maxprot);
          if ((segCmd->initprot != 0) && ((segCmd->initprot & VM_PROT_READ) == 0))
            tvm::throwf_("malformed mach-o image: %s segment is not mapped readable",
                         segCmd->segname);
        }
        if ((segCmd->fileoff == 0) && (segCmd->filesize != 0)) {
          if ((segCmd->initprot & VM_PROT_READ) == 0)
            tvm::throwf_(
                "malformed mach-o image: %s segment maps start of file but is not readable",
                segCmd->segname);
          if ((segCmd->initprot & VM_PROT_WRITE) == VM_PROT_WRITE) {
            if (context.strictMachORequired)
              tvm::throwf_("malformed mach-o image: %s segment maps start of file but is writable",
                           segCmd->segname);
          }
          if (segCmd->filesize < (sizeof(macho_header) + mh->sizeofcmds))
            tvm::throwf_("malformed mach-o image: %s segment does not map all of load commands",
                         segCmd->segname);
          if (startOfFileSegCmd != NULL)
            tvm::throwf_("malformed mach-o image: multiple segments map start of file: %s %s",
                         startOfFileSegCmd->segname, segCmd->segname);
          startOfFileSegCmd = segCmd;
        }
        if (context.strictMachORequired) {
          uintptr_t vmStart = segCmd->vmaddr;
          uintptr_t vmSize = segCmd->vmsize;
          uintptr_t vmEnd = vmStart + vmSize;
          uintptr_t fileStart = segCmd->fileoff;
          uintptr_t fileSize = segCmd->filesize;
          if ((intptr_t)(vmSize) < 0)
            tvm::throwf_("malformed mach-o image: segment load command %s vmsize too large in %s",
                         segCmd->segname, path);
          if (vmStart > vmEnd)
            tvm::throwf_(
                "malformed mach-o image: segment load command %s wraps around address space",
                segCmd->segname);
          if (vmSize != fileSize) {
            if (segCmd->initprot == 0) {
              // allow: fileSize == 0 && initprot == 0		e.g. __PAGEZERO
              // allow: vmSize == 0 && initprot == 0			e.g. __LLVM
              if ((fileSize != 0) && (vmSize != 0))
                tvm::throwf_(
                    "malformed mach-o image: unaccessable segment %s has non-zero filesize and vmsize",
                    segCmd->segname);
            } else {
              // allow: vmSize > fileSize && initprot != X  e.g. __DATA
              if (vmSize < fileSize) {
                tvm::throwf_("malformed mach-o image: segment %s has vmsize < filesize",
                             segCmd->segname);
              }
              if (segCmd->initprot & VM_PROT_EXECUTE) {
                tvm::throwf_(
                    "malformed mach-o image: segment %s has vmsize != filesize and is executable",
                    segCmd->segname);
              }
            }
          }
          if (inCache) {
            if ((fileSize != 0) && (segCmd->initprot == (VM_PROT_READ | VM_PROT_EXECUTE))) {
              if (foundLoadCommandSegment) throw "load commands in multiple segments";
              foundLoadCommandSegment = true;
            }
          } else if ((fileStart < mh->sizeofcmds) && (fileSize != 0)) {
            // <rdar://problem/7942521> all load commands must be in an executable segment
            if ((fileStart != 0) || (fileSize < (mh->sizeofcmds + sizeof(macho_header))))
              tvm::throwf_("malformed mach-o image: segment %s does not span all load commands",
                           segCmd->segname);
            if (segCmd->initprot != (VM_PROT_READ | VM_PROT_EXECUTE))
              tvm::throwf_(
                  "malformed mach-o image: load commands found in segment %s with wrong permissions",
                  segCmd->segname);
            if (foundLoadCommandSegment) throw "load commands in multiple segments";
            foundLoadCommandSegment = true;
          }

          const macho_section* const sectionsStart =
              (macho_section*)((char*)segCmd + sizeof(macho_segment_command));
          const macho_section* const sectionsEnd = &sectionsStart[segCmd->nsects];
          for (const macho_section* sect = sectionsStart; sect < sectionsEnd; ++sect) {
            if (!inCache && sect->offset != 0 &&
                ((sect->offset + sect->size) > (segCmd->fileoff + segCmd->filesize)))
              tvm::throwf_(
                  "malformed mach-o image: section %s,%s of '%s' exceeds segment %s booundary",
                  sect->segname, sect->sectname, path, segCmd->segname);
          }
        }
        break;
      case LC_SEGMENT_COMMAND_WRONG:
        tvm::throwf_("malformed mach-o image: wrong LC_SEGMENT[_64] for architecture");
        break;
      case LC_LOAD_DYLIB:
      case LC_LOAD_WEAK_DYLIB:
      case LC_REEXPORT_DYLIB:
      case LC_LOAD_UPWARD_DYLIB:
        *libCount += 1;
        // fall thru
        [[clang::fallthrough]];
      case LC_ID_DYLIB:
        dylibCmd = (dylib_command*)cmd;
        if (dylibCmd->dylib.name.offset > cmdLength)
          tvm::throwf_(
              "malformed mach-o image: dylib load command #%d has offset (%u) outside its size (%u)",
              i, dylibCmd->dylib.name.offset, cmdLength);
        if ((dylibCmd->dylib.name.offset + strlen((char*)dylibCmd + dylibCmd->dylib.name.offset) +
             1) > cmdLength)
          tvm::throwf_(
              "malformed mach-o image: dylib load command #%d string extends beyond end of load command",
              i);
        break;
      case LC_CODE_SIGNATURE:
        if (cmd->cmdsize != sizeof(linkedit_data_command))
          throw "malformed mach-o image: LC_CODE_SIGNATURE size wrong";
        // <rdar://problem/22799652> only support one LC_CODE_SIGNATURE per image
        if (*codeSigCmd != NULL)
          throw "malformed mach-o image: multiple LC_CODE_SIGNATURE load commands";
        *codeSigCmd = (struct linkedit_data_command*)cmd;
        break;
      case LC_ENCRYPTION_INFO:
        if (cmd->cmdsize != sizeof(encryption_info_command))
          throw "malformed mach-o image: LC_ENCRYPTION_INFO size wrong";
        // <rdar://problem/22799652> only support one LC_ENCRYPTION_INFO per image
        if (*encryptCmd != NULL)
          throw "malformed mach-o image: multiple LC_ENCRYPTION_INFO load commands";
        *encryptCmd = (encryption_info_command*)cmd;
        break;
      case LC_ENCRYPTION_INFO_64:
        if (cmd->cmdsize != sizeof(encryption_info_command_64))
          throw "malformed mach-o image: LC_ENCRYPTION_INFO_64 size wrong";
        // <rdar://problem/22799652> only support one LC_ENCRYPTION_INFO_64 per image
        if (*encryptCmd != NULL)
          throw "malformed mach-o image: multiple LC_ENCRYPTION_INFO_64 load commands";
        *encryptCmd = (encryption_info_command*)cmd;
        break;
      case LC_SYMTAB:
        if (cmd->cmdsize != sizeof(symtab_command))
          throw "malformed mach-o image: LC_SYMTAB size wrong";
        symTabCmd = (symtab_command*)cmd;
        break;
      case LC_DYSYMTAB:
        if (cmd->cmdsize != sizeof(dysymtab_command))
          throw "malformed mach-o image: LC_DYSYMTAB size wrong";
        dynSymbTabCmd = (dysymtab_command*)cmd;
        break;
    }
    cmd = nextCmd;
  }

  if (context.strictMachORequired && !foundLoadCommandSegment)
    throw "load commands not in a segment";
  if (linkeditSegCmd == NULL) throw "malformed mach-o image: missing __LINKEDIT segment";
  if (!inCache && (startOfFileSegCmd == NULL))
    throw "malformed mach-o image: missing __TEXT segment that maps start of file";
  // <rdar://problem/13145644> verify every segment does not overlap another segment
  if (context.strictMachORequired) {
    uintptr_t lastFileStart = 0;
    uintptr_t linkeditFileStart = 0;
    const struct load_command* cmd1 = startCmds;
    for (uint32_t i = 0; i < cmd_count; ++i) {
      if (cmd1->cmd == LC_SEGMENT_COMMAND) {
        macho_segment_command* segCmd1 = (macho_segment_command*)cmd1;
        uintptr_t vmStart1 = segCmd1->vmaddr;
        uintptr_t vmEnd1 = segCmd1->vmaddr + segCmd1->vmsize;
        uintptr_t fileStart1 = segCmd1->fileoff;
        uintptr_t fileEnd1 = segCmd1->fileoff + segCmd1->filesize;

        if (fileStart1 > lastFileStart) lastFileStart = fileStart1;

        if (strcmp(&segCmd1->segname[0], "__LINKEDIT") == 0) {
          linkeditFileStart = fileStart1;
        }

        const struct load_command* cmd2 = startCmds;
        for (uint32_t j = 0; j < cmd_count; ++j) {
          if (cmd2 == cmd1) continue;
          if (cmd2->cmd == LC_SEGMENT_COMMAND) {
            macho_segment_command* segCmd2 = (macho_segment_command*)cmd2;
            uintptr_t vmStart2 = segCmd2->vmaddr;
            uintptr_t vmEnd2 = segCmd2->vmaddr + segCmd2->vmsize;
            uintptr_t fileStart2 = segCmd2->fileoff;
            uintptr_t fileEnd2 = segCmd2->fileoff + segCmd2->filesize;
            if (((vmStart2 <= vmStart1) && (vmEnd2 > vmStart1) && (vmEnd1 > vmStart1)) ||
                ((vmStart2 >= vmStart1) && (vmStart2 < vmEnd1) && (vmEnd2 > vmStart2)))
              tvm::throwf_("malformed mach-o image: segment %s vm overlaps segment %s",
                           segCmd1->segname, segCmd2->segname);
            if (((fileStart2 <= fileStart1) && (fileEnd2 > fileStart1) &&
                 (fileEnd1 > fileStart1)) ||
                ((fileStart2 >= fileStart1) && (fileStart2 < fileEnd1) && (fileEnd2 > fileStart2)))
              tvm::throwf_("malformed mach-o image: segment %s file content overlaps segment %s",
                           segCmd1->segname, segCmd2->segname);
          }
          cmd2 = (const struct load_command*)(((char*)cmd2) + cmd2->cmdsize);
        }
      }
      cmd1 = (const struct load_command*)(((char*)cmd1) + cmd1->cmdsize);
    }

    if (lastFileStart != linkeditFileStart)
      tvm::throwf_("malformed mach-o image: __LINKEDIT must be last segment");
  }

  // validate linkedit content
  if ((dyldInfoCmd == NULL) && (chainedFixupsCmd == NULL) && (symTabCmd == NULL))
    throw "malformed mach-o image: missing LC_SYMTAB, LC_DYLD_INFO, or LC_DYLD_CHAINED_FIXUPS";
  if (dynSymbTabCmd == NULL) throw "malformed mach-o image: missing LC_DYSYMTAB";

  uint32_t linkeditFileOffsetStart = (uint32_t)linkeditSegCmd->fileoff;
  uint32_t linkeditFileOffsetEnd =
      (uint32_t)linkeditSegCmd->fileoff + (uint32_t)linkeditSegCmd->filesize;

  if (!inCache && (dyldInfoCmd != NULL) && context.strictMachORequired) {
    // validate all LC_DYLD_INFO chunks fit in LINKEDIT and don't overlap
    uint32_t offset = linkeditFileOffsetStart;
    if (dyldInfoCmd->rebase_size != 0) {
      if (dyldInfoCmd->rebase_size & 0x80000000)
        throw "malformed mach-o image: dyld rebase info size overflow";
      if (dyldInfoCmd->rebase_off < offset)
        throw "malformed mach-o image: dyld rebase info underruns __LINKEDIT";
      offset = dyldInfoCmd->rebase_off + dyldInfoCmd->rebase_size;
      if (offset > linkeditFileOffsetEnd)
        throw "malformed mach-o image: dyld rebase info overruns __LINKEDIT";
    }
    if (dyldInfoCmd->bind_size != 0) {
      if (dyldInfoCmd->bind_size & 0x80000000)
        throw "malformed mach-o image: dyld bind info size overflow";
      if (dyldInfoCmd->bind_off < offset)
        throw "malformed mach-o image: dyld bind info overlaps rebase info";
      offset = dyldInfoCmd->bind_off + dyldInfoCmd->bind_size;
      if (offset > linkeditFileOffsetEnd)
        throw "malformed mach-o image: dyld bind info overruns __LINKEDIT";
    }
    if (dyldInfoCmd->weak_bind_size != 0) {
      if (dyldInfoCmd->weak_bind_size & 0x80000000)
        throw "malformed mach-o image: dyld weak bind info size overflow";
      if (dyldInfoCmd->weak_bind_off < offset)
        throw "malformed mach-o image: dyld weak bind info overlaps bind info";
      offset = dyldInfoCmd->weak_bind_off + dyldInfoCmd->weak_bind_size;
      if (offset > linkeditFileOffsetEnd)
        throw "malformed mach-o image: dyld weak bind info overruns __LINKEDIT";
    }
    if (dyldInfoCmd->lazy_bind_size != 0) {
      if (dyldInfoCmd->lazy_bind_size & 0x80000000)
        throw "malformed mach-o image: dyld lazy bind info size overflow";
      if (dyldInfoCmd->lazy_bind_off < offset)
        throw "malformed mach-o image: dyld lazy bind info overlaps weak bind info";
      offset = dyldInfoCmd->lazy_bind_off + dyldInfoCmd->lazy_bind_size;
      if (offset > linkeditFileOffsetEnd)
        throw "malformed mach-o image: dyld lazy bind info overruns __LINKEDIT";
    }
    if (dyldInfoCmd->export_size != 0) {
      if (dyldInfoCmd->export_size & 0x80000000)
        throw "malformed mach-o image: dyld export info size overflow";
      if (dyldInfoCmd->export_off < offset)
        throw "malformed mach-o image: dyld export info overlaps lazy bind info";
      offset = dyldInfoCmd->export_off + dyldInfoCmd->export_size;
      if (offset > linkeditFileOffsetEnd)
        throw "malformed mach-o image: dyld export info overruns __LINKEDIT";
    }
  }

  if (!inCache && (chainedFixupsCmd != NULL) && context.strictMachORequired) {
    // validate all LC_DYLD_CHAINED_FIXUPS chunks fit in LINKEDIT and don't overlap
    if (chainedFixupsCmd->dataoff < linkeditFileOffsetStart)
      throw "malformed mach-o image: dyld chained fixups info underruns __LINKEDIT";
    if ((chainedFixupsCmd->dataoff + chainedFixupsCmd->datasize) > linkeditFileOffsetEnd)
      throw "malformed mach-o image: dyld chained fixups info overruns __LINKEDIT";
  }

  if (!inCache && (exportsTrieCmd != NULL) && context.strictMachORequired) {
    // validate all LC_DYLD_EXPORTS_TRIE chunks fit in LINKEDIT and don't overlap
    if (exportsTrieCmd->dataoff < linkeditFileOffsetStart)
      throw "malformed mach-o image: dyld chained fixups info underruns __LINKEDIT";
    if ((exportsTrieCmd->dataoff + exportsTrieCmd->datasize) > linkeditFileOffsetEnd)
      throw "malformed mach-o image: dyld chained fixups info overruns __LINKEDIT";
  }

  if (symTabCmd != NULL) {
    // validate symbol table fits in LINKEDIT
    if ((symTabCmd->nsyms > 0) && (symTabCmd->symoff < linkeditFileOffsetStart))
      throw "malformed mach-o image: symbol table underruns __LINKEDIT";
    if (symTabCmd->nsyms > 0x10000000) throw "malformed mach-o image: symbol table too large";
    uint32_t symbolsSize = symTabCmd->nsyms * sizeof(macho_nlist);
    if (symbolsSize > linkeditSegCmd->filesize)
      throw "malformed mach-o image: symbol table overruns __LINKEDIT";
    if (symTabCmd->symoff + symbolsSize < symTabCmd->symoff)
      throw "malformed mach-o image: symbol table size wraps";
    if (symTabCmd->symoff + symbolsSize > symTabCmd->stroff)
      throw "malformed mach-o image: symbol table overlaps symbol strings";
    if (symTabCmd->stroff + symTabCmd->strsize < symTabCmd->stroff)
      throw "malformed mach-o image: symbol string size wraps";
    if (symTabCmd->stroff + symTabCmd->strsize > linkeditFileOffsetEnd) {
      // <rdar://problem/24220313> let old apps overflow as long as it stays within mapped page
      if (context.strictMachORequired ||
          (symTabCmd->stroff + symTabCmd->strsize > ((linkeditFileOffsetEnd + 4095) & (-4096))))
        throw "malformed mach-o image: symbol strings overrun __LINKEDIT";
    }
    // validate indirect symbol table
    if (dynSymbTabCmd->nindirectsyms != 0) {
      if (dynSymbTabCmd->indirectsymoff < linkeditFileOffsetStart)
        throw "malformed mach-o image: indirect symbol table underruns __LINKEDIT";
      if (dynSymbTabCmd->nindirectsyms > 0x10000000)
        throw "malformed mach-o image: indirect symbol table too large";
      uint32_t indirectTableSize = dynSymbTabCmd->nindirectsyms * sizeof(uint32_t);
      if (indirectTableSize > linkeditSegCmd->filesize)
        throw "malformed mach-o image: indirect symbol table overruns __LINKEDIT";
      if (dynSymbTabCmd->indirectsymoff + indirectTableSize < dynSymbTabCmd->indirectsymoff)
        throw "malformed mach-o image: indirect symbol table size wraps";
      if (context.strictMachORequired &&
          (dynSymbTabCmd->indirectsymoff + indirectTableSize > symTabCmd->stroff))
        throw "malformed mach-o image: indirect symbol table overruns string pool";
    }
    if ((dynSymbTabCmd->nlocalsym > symTabCmd->nsyms) ||
        (dynSymbTabCmd->ilocalsym > symTabCmd->nsyms))
      throw "malformed mach-o image: indirect symbol table local symbol count exceeds total symbols";
    if (dynSymbTabCmd->ilocalsym + dynSymbTabCmd->nlocalsym < dynSymbTabCmd->ilocalsym)
      throw "malformed mach-o image: indirect symbol table local symbol count wraps";
    if ((dynSymbTabCmd->nextdefsym > symTabCmd->nsyms) ||
        (dynSymbTabCmd->iextdefsym > symTabCmd->nsyms))
      throw "malformed mach-o image: indirect symbol table extern symbol count exceeds total symbols";
    if (dynSymbTabCmd->iextdefsym + dynSymbTabCmd->nextdefsym < dynSymbTabCmd->iextdefsym)
      throw "malformed mach-o image: indirect symbol table extern symbol count wraps";
    if ((dynSymbTabCmd->nundefsym > symTabCmd->nsyms) ||
        (dynSymbTabCmd->iundefsym > symTabCmd->nsyms))
      throw "malformed mach-o image: indirect symbol table undefined symbol count exceeds total symbols";
    if (dynSymbTabCmd->iundefsym + dynSymbTabCmd->nundefsym < dynSymbTabCmd->iundefsym)
      throw "malformed mach-o image: indirect symbol table undefined symbol count wraps";
  }

  // fSegmentsArrayCount is only 8-bits
  if (*segCount > 255) tvm::throwf_("malformed mach-o image: more than 255 segments in %s", path);

  // fSegmentsArrayCount is only 8-bits
  if (*libCount > 4095)
    tvm::throwf_("malformed mach-o image: more than 4095 dependent libraries in %s", path);

  if (needsAddedLibSystemDepency(*libCount, mh)) *libCount = 1;

  // dylibs that use LC_DYLD_CHAINED_FIXUPS have that load command removed when put in the dyld cache
  if (!*compressed && (mh->flags & MH_DYLIB_IN_CACHE)) *compressed = true;
}

void unsafe_loader::parseLoadCmds(const LinkContext& context)
{
  // now that segments are mapped in, get real fMachOData, fLinkEditBase, and fSlide
  for(unsigned int i=0; i < fSegmentsCount; ++i) {
    // set up pointer to __LINKEDIT segment
    if ( strcmp(segName(i),"__LINKEDIT") == 0 ) {
      fLinkEditBase = (uint8_t*)(segActualLoadAddress(i) - segFileOffset(i));
    }
    // some segment always starts at beginning of file and contains mach_header and load commands
    if ( (segFileOffset(i) == 0) && (segFileSize(i) != 0) ) {
      fMachOData = (uint8_t*)(segActualLoadAddress(i));
    }
  }

  // walk load commands (mapped in at start of __TEXT segment)
  const dyld_info_command* dyldInfo = NULL;
  const linkedit_data_command* chainedFixupsCmd = NULL;
  const linkedit_data_command* exportsTrieCmd = NULL;
  const macho_nlist* symbolTable = NULL;
  const char* symbolTableStrings = NULL;
  const struct load_command* firstUnknownCmd = NULL;
  const struct version_min_command* minOSVersionCmd = NULL;
  const dysymtab_command* dynSymbolTable = NULL;
  const uint32_t cmd_count = ((macho_header*)fMachOData)->ncmds;
  const struct load_command* const cmds = (struct load_command*)&fMachOData[sizeof(macho_header)];
  const struct load_command* cmd = cmds;
  for (uint32_t i = 0; i < cmd_count; ++i) {
    switch (cmd->cmd) {
      case LC_SYMTAB:
      {
        const struct symtab_command* symtab = (struct symtab_command*)cmd;
        symbolTableStrings = (const char*)&fLinkEditBase[symtab->stroff];
        symbolTable = (macho_nlist*)(&fLinkEditBase[symtab->symoff]);
      }
        break;
      case LC_DYSYMTAB:
        dynSymbolTable = (struct dysymtab_command*)cmd;
        break;
      case LC_SUB_UMBRELLA:
        fHasSubUmbrella = true;
        break;
      case LC_SUB_FRAMEWORK:
        fInUmbrella = true;
        break;
      case LC_SUB_LIBRARY:
        fHasSubLibraries = true;
        break;
      case LC_ROUTINES_COMMAND:
        fHasDashInit = true;
        break;
      case LC_DYLD_INFO:
      case LC_DYLD_INFO_ONLY:
        dyldInfo = (struct dyld_info_command*)cmd;
        break;
      case LC_DYLD_CHAINED_FIXUPS:
        chainedFixupsCmd = (struct linkedit_data_command*)cmd;
        break;
      case LC_DYLD_EXPORTS_TRIE:
        exportsTrieCmd = (struct linkedit_data_command*)cmd;
        break;
      case LC_SEGMENT_COMMAND:
      {
        const macho_segment_command* seg = (macho_segment_command*)cmd;
        const bool isTextSeg = (strcmp(seg->segname, "__TEXT") == 0);
#if __i386__ && TARGET_OS_OSX
        const bool isObjCSeg = (strcmp(seg->segname, "__OBJC") == 0);
					if ( isObjCSeg )
						fNotifyObjC = true;
#else
        const bool isDataSeg = (strncmp(seg->segname, "__DATA", 6) == 0);
#endif
        const macho_section* const sectionsStart = (macho_section*)((char*)seg + sizeof(macho_segment_command));
        const macho_section* const sectionsEnd = &sectionsStart[seg->nsects];
        for (const macho_section* sect=sectionsStart; sect < sectionsEnd; ++sect) {
          const uint8_t type = sect->flags & SECTION_TYPE;
          if ( type == S_MOD_INIT_FUNC_POINTERS )
            fHasInitializers = true;
          else if ( type == S_INIT_FUNC_OFFSETS )
            fHasInitializers = true;
          else if ( type == S_MOD_TERM_FUNC_POINTERS )
            fHasTerminators = true;
          else if ( type == S_DTRACE_DOF )
            fHasDOFSections = true;
          else if ( isTextSeg && (strcmp(sect->sectname, "__eh_frame") == 0) )
            fEHFrameSectionOffset = (uint32_t)((uint8_t*)sect - fMachOData);
          else if ( isTextSeg && (strcmp(sect->sectname, "__unwind_info") == 0) )
            fUnwindInfoSectionOffset = (uint32_t)((uint8_t*)sect - fMachOData);
        }
      }
        break;
      case LC_TWOLEVEL_HINTS:
        // no longer supported
        break;
      case LC_ID_DYLIB:
      {
        fDylibIDOffset = (uint32_t)((uint8_t*)cmd - fMachOData);
      }
        break;
      case LC_RPATH:
      case LC_LOAD_WEAK_DYLIB:
      case LC_REEXPORT_DYLIB:
      case LC_LOAD_UPWARD_DYLIB:
      case LC_MAIN:
        break;
      case LC_VERSION_MIN_MACOSX:
      case LC_VERSION_MIN_IPHONEOS:
      case LC_VERSION_MIN_TVOS:
      case LC_VERSION_MIN_WATCHOS:
        minOSVersionCmd = (version_min_command*)cmd;
        break;
      default:
        if ( (cmd->cmd & LC_REQ_DYLD) != 0 ) {
          if ( firstUnknownCmd == NULL )
            firstUnknownCmd = cmd;
        }
        break;
    }
    cmd = (const struct load_command*)(((char*)cmd)+cmd->cmdsize);
  }
  if ( firstUnknownCmd != NULL ) {
    if ( minOSVersionCmd != NULL )  {
      tvm::throwf_("cannot load because it was built for OS version %u.%u (load command 0x%08X is unknown)",
                   minOSVersionCmd->version >> 16, ((minOSVersionCmd->version >> 8) & 0xff),
                   firstUnknownCmd->cmd);
    }
    else {
      tvm::throwf_("cannot load (load command 0x%08X is unknown)", firstUnknownCmd->cmd);
    }
  }


  if ( dyldInfo != NULL )
    this->fDyldInfo = dyldInfo;
  if ( chainedFixupsCmd != NULL )
    tvm::throwf_("Loader unimplemented");
//    this->setChainedFixups(chainedFixupsCmd);
  if ( exportsTrieCmd != NULL )
    tvm::throwf_("Loader unimplemented");
//    this->setExportsTrie(exportsTrieCmd);

//  if ( symbolTable != NULL)
//    this->setSymbolTableInfo(symbolTable, symbolTableStrings, dynSymbolTable);
}


bool unsafe_loader::needsAddedLibSystemDepency(unsigned int libCount, const macho_header* mh)
{
  // <rdar://problem/6357561> ensure that every image depends on something which depends on libSystem
  if ( libCount > 1 )
    return false;

  // <rdar://problem/6409800> dyld implicit-libSystem breaks valgrind
  if ( mh->filetype == MH_EXECUTE )
    return false;

  bool isNonOSdylib = false;
  const uint32_t cmd_count = mh->ncmds;
  const struct load_command* const cmds = (struct load_command*)((uint8_t*)mh+sizeof(macho_header));
  const struct load_command* cmd = cmds;
  for (uint32_t i = 0; i < cmd_count; ++i) {
    switch (cmd->cmd) {
      case LC_LOAD_DYLIB:
      case LC_LOAD_WEAK_DYLIB:
      case LC_REEXPORT_DYLIB:
      case LC_LOAD_UPWARD_DYLIB:
        return false;
      case LC_ID_DYLIB:
      {
        const dylib_command* dylibID = (dylib_command*)cmd;
        const char* installPath = (char*)cmd + dylibID->dylib.name.offset;
        // It is OK for OS dylibs (libSystem or libmath) to have no dependents
        // but all other dylibs must depend on libSystem for initialization to initialize libSystem first
        isNonOSdylib = ( (strncmp(installPath, "/usr/lib/", 9) != 0) && (strncmp(installPath, "/System/DriverKit/usr/lib/", 26) != 0) );
        // if (isNonOSdylib) tvm::log_("unsafe_loader::needsAddedLibSystemDepency(%s)\n", installPath);
      }
        break;
    }
    cmd = (const struct load_command*)(((char*)cmd)+cmd->cmdsize);
  }
  return isNonOSdylib;
}


unsafe_loader* unsafe_loader::instantiateFromMemory(const char* moduleName, const macho_header* mh, uint64_t len,
                                                    unsigned int segCount, unsigned int libCount, const LinkContext& context)
{
  auto* image = unsafe_loader::instantiateStart(mh, moduleName, segCount, libCount);
  try {
    // map segments
    if ( mh->filetype == MH_EXECUTE )
      throw "can't load another MH_EXECUTE";

    // vmcopy segments
    image->mapSegments((const void*)mh, len, context);
#ifndef TVM_FOR
    // for compatibility, never unload dylibs loaded from memory
		image->setNeverUnload();
    image->disableCoverageCheck();

    // bundle loads need path copied
    if ( moduleName != NULL )
      image->setPath(moduleName);

    image->instantiateFinish(context);
    image->setMapped(context);
#endif
    image->parseLoadCmds(context);
  }
  catch (...) {
    // ImageLoader::setMapped() can throw an exception to block loading of image
    // <rdar://problem/6169686> Leaked fSegmentsArray and image segments during failed dlopen_preflight
    delete image;
    throw;
  }

  return image;
}

// construct unsafe_loaderCompressed using "placement new" with SegmentMachO objects array at end
unsafe_loader* unsafe_loader::instantiateStart(const macho_header* mh, const char* path,
                                               unsigned int segCount, unsigned int libCount)
{
  size_t size = sizeof(unsafe_loader) + segCount * sizeof(uint32_t) + libCount * sizeof(unsafe_loader*);
  unsafe_loader* allocatedSpace = static_cast<unsafe_loader*>(malloc(size));
  if ( allocatedSpace == NULL )
    throw "malloc failed";
  uint32_t* segOffsets = ((uint32_t*)(((uint8_t*)allocatedSpace) + sizeof(unsafe_loader)));
  bzero(&segOffsets[segCount], libCount*sizeof(void*));	// zero out lib array
  return new (allocatedSpace) unsafe_loader(mh, path, segCount, segOffsets, libCount);
}

unsafe_loader::unsafe_loader(const macho_header* mh, const char* path, unsigned int segCount,
                            uint32_t segOffsets[], unsigned int libCount)
    :
//      fPath(path),
//      fLibraryCount(libCount),
      fDyldInfo(NULL),
      fChainedFixups(NULL),
      fExportsTrie(NULL),
//      fCoveredCodeLength(0),
      fMachOData((uint8_t*)mh),
      fLinkEditBase(NULL),
      fSlide(0),
//      fEHFrameSectionOffset(0),
//      fUnwindInfoSectionOffset(0),
//      fDylibIDOffset(0),
//      fReadOnlyDataSegment(false),
//      fHasSubLibraries(false), fHasSubUmbrella(false), fInUmbrella(false), fHasDOFSections(false), fHasDashInit(false),
//      fHasInitializers(false), fHasTerminators(false), fNotifyObjC(false), fRetainForObjC(false), fRegisteredAsRequiresCoalescing(false), fOverrideOfCacheImageNum(0)
      fSegmentsCount(segCount), fIsSplitSeg(false), fInSharedCache(false)
{
  fIsSplitSeg = ((mh->flags & MH_SPLIT_SEGS) != 0);

  // construct SegmentMachO object for each LC_SEGMENT cmd using "placement new" to put
  // each SegmentMachO object in array at end of unsafe_loader object
  const uint32_t cmd_count = mh->ncmds;
  const struct load_command* const cmds = (struct load_command*)&fMachOData[sizeof(macho_header)];
  const struct load_command* cmd = cmds;
  for (uint32_t i = 0, segIndex=0; i < cmd_count; ++i) {
    if ( cmd->cmd == LC_SEGMENT_COMMAND ) {
      const macho_segment_command* segCmd = (macho_segment_command*)cmd;
      // ignore zero-sized segments
      if ( segCmd->vmsize != 0 ) {
        // record offset of load command
        segOffsets[segIndex++] = (uint32_t)((uint8_t*)segCmd - fMachOData);
      }
    }
    cmd = (const struct load_command*)(((char*)cmd)+cmd->cmdsize);
  }
}

void unsafe_loader::mapSegments(const void* memoryImage, uint64_t imageLen, const LinkContext& context)
{
  // find address range for image
  intptr_t slide = this->assignSegmentAddresses(context, 0);
  if ( context.verboseMapping )
    tvm::log_("dyld: Mapping memory %p\n", memoryImage);
  // map in all segments
  for(unsigned int i=0, e=fSegmentsCount; i < e; ++i) {
    vm_address_t loadAddress = segPreferredLoadAddress(i) + slide;
    vm_address_t srcAddr = (uintptr_t)memoryImage + segFileOffset(i);
    vm_size_t size = segFileSize(i);
//		kern_return_t r = vm_copy(mach_task_self(), srcAddr, size, loadAddress);
    memcpy(reinterpret_cast<void*>(loadAddress), reinterpret_cast<void*>(srcAddr), size);
    kern_return_t r = KERN_SUCCESS;
    if ( r != KERN_SUCCESS )
      throw "can't map segment";
    if ( context.verboseMapping )
      tvm::log_("%18s at 0x%08lX->0x%08lX\n", segName(i), (uintptr_t)loadAddress, (uintptr_t)loadAddress+size-1);
  }
  // update slide to reflect load location
  this->fSlide = slide;
  // set R/W permissions on all segments at slide location
  for(unsigned int i=0, e=fSegmentsCount; i < e; ++i) {
//		segProtect(i, context);
  }
}

intptr_t unsafe_loader::assignSegmentAddresses(const LinkContext& context, size_t extraAllocationSize) {
  uintptr_t lowAddr = (unsigned long)(-1);
  uintptr_t highAddr = 0;

  intptr_t slide = 0;
  bool needsToSlide = false;
  intptr_t segmentReAlignSlide = 0;

  for (unsigned int i = 0, e = fSegmentsCount; i < e; ++i) {
    const uintptr_t segLow = segPreferredLoadAddress(i);
    const uintptr_t segHigh = dyld_page_round(segLow + segSize(i));
    if (segLow < highAddr) {
      if (dyld_page_size > 4096)
        tvm::throwf_("can't map segments into 16KB pages");
      else
        tvm::throwf_("overlapping segments");
    }
    if (segLow < lowAddr) lowAddr = segLow;
    if (segHigh > highAddr) highAddr = segHigh;
  }

  // find a chunk of address space to hold all segments
  size_t size = highAddr-lowAddr+segmentReAlignSlide;
  uintptr_t addr = reinterpret_cast<uintptr_t>(malloc(size+extraAllocationSize));
  slide = addr - lowAddr + segmentReAlignSlide;

  return slide;

//  // preflight and calculate slide if needed
//  const bool inPIE = (fgNextPIEDylibAddress != 0);
//  intptr_t slide = 0;
//  if ( this->segmentsCanSlide() && this->segmentsMustSlideTogether() ) {
//    intptr_t segmentReAlignSlide = 0;
//    bool needsToSlide = false;
//    bool imageHasPreferredLoadAddress = segHasPreferredLoadAddress(0);
//    uintptr_t lowAddr = (unsigned long)(-1);
//    uintptr_t highAddr = 0;
//    for(unsigned int i=0, e=segmentCount(); i < e; ++i) {
//      const uintptr_t segLow = segPreferredLoadAddress(i);
//      const uintptr_t segHigh = dyld_page_round(segLow + segSize(i));
//      if ( segLow < highAddr ) {
//        if ( dyld_page_size > 4096 )
//          tvm::throwf_("can't map segments into 16KB pages");
//        else
//          tvm::throwf_("overlapping segments");
//      }
//      if ( segLow < lowAddr )
//        lowAddr = segLow;
//      if ( segHigh > highAddr )
//        highAddr = segHigh;
//
//      if ( needsToSlide || !imageHasPreferredLoadAddress || inPIE || !reserveAddressRange(segPreferredLoadAddress(i), segSize(i)) )
//        needsToSlide = true;
//    }
//    if ( needsToSlide ) {
//      // find a chunk of address space to hold all segments
//      size_t size = highAddr-lowAddr+segmentReAlignSlide;
////			uintptr_t addr = reserveAnAddressRange(size+extraAllocationSize, context);
//      uintptr_t addr = reinterpret_cast<uintptr_t>(malloc(size+extraAllocationSize));
//      slide = addr - lowAddr + segmentReAlignSlide;
//    } else if ( extraAllocationSize ) {
//      if (!reserveAddressRange(highAddr, extraAllocationSize)) {
//        throw "failed to reserve space for aot";
//      }
//    }
//  }
//  else if ( ! this->segmentsCanSlide() ) {
//    uintptr_t highAddr = 0;
//    for(unsigned int i=0, e=fSegmentsCount; i < e; ++i) {
//      const uintptr_t segLow = segPreferredLoadAddress(i);
//      const uintptr_t segHigh = dyld_page_round(segLow + segSize(i));
//
//      if ( segHigh > highAddr )
//        highAddr = segHigh;
//
//      if ( (strcmp(segName(i), "__PAGEZERO") == 0) && (segFileSize(i) == 0) && (segPreferredLoadAddress(i) == 0) )
//        continue;
//      if ( !reserveAddressRange(segPreferredLoadAddress(i), segSize(i)) )
//        tvm::throwf_("can't map unslidable segment %s to 0x%lX with size 0x%lX", segName(i), segPreferredLoadAddress(i), segSize(i));
//    }
//    if (extraAllocationSize) {
//      tvm::throwf_("binaries with non-slidable segments don't support aot: %s", this->getPath());
//    }
//  }
//  else {
//    throw "mach-o does not support independently sliding segments";
//  }
//  return slide;
}

uint32_t* unsafe_loader::segmentCommandOffsets() const
{
  return ((uint32_t*)(((uint8_t*)this) + sizeof(unsafe_loader)));
}

const macho_segment_command* unsafe_loader::segLoadCommand(unsigned int segIndex) const
{
  uint32_t* lcOffsets = this->segmentCommandOffsets();
  uint32_t lcOffset =	lcOffsets[segIndex];
  return (macho_segment_command*)(&fMachOData[lcOffset]);
}

const char* unsafe_loader::segName(unsigned int segIndex) const
{
  return segLoadCommand(segIndex)->segname;
}

uintptr_t unsafe_loader::segSize(unsigned int segIndex) const
{
  return segLoadCommand(segIndex)->vmsize;
}


uintptr_t unsafe_loader::segFileSize(unsigned int segIndex) const
{
  return segLoadCommand(segIndex)->filesize;
}


//bool unsafe_loader::segHasTrailingZeroFill(unsigned int segIndex)
//{
//  return ( segWriteable(segIndex) && (segSize(segIndex) > segFileSize(segIndex)) );
//}


uintptr_t unsafe_loader::segFileOffset(unsigned int segIndex) const
{
  return segLoadCommand(segIndex)->fileoff;
}


//bool unsafe_loader::segReadable(unsigned int segIndex) const
//{
//  return ( (segLoadCommand(segIndex)->initprot & VM_PROT_READ) != 0);
//}


//bool unsafe_loader::segWriteable(unsigned int segIndex) const
//{
//  return ( (segLoadCommand(segIndex)->initprot & VM_PROT_WRITE) != 0);
//}


//bool unsafe_loader::segExecutable(unsigned int segIndex) const
//{
//  return ( (segLoadCommand(segIndex)->initprot & VM_PROT_EXECUTE) != 0);
//}


//bool unsafe_loader::segUnaccessible(unsigned int segIndex) const
//{
//  return (segLoadCommand(segIndex)->initprot == 0);
//}

bool unsafe_loader::segHasPreferredLoadAddress(unsigned int segIndex) const
{
  return (segLoadCommand(segIndex)->vmaddr != 0);
}

uintptr_t unsafe_loader::segPreferredLoadAddress(unsigned int segIndex) const
{
  return segLoadCommand(segIndex)->vmaddr;
}

uintptr_t unsafe_loader::segActualLoadAddress(unsigned int segIndex) const
{
  return segLoadCommand(segIndex)->vmaddr + fSlide;
}


uintptr_t unsafe_loader::segActualEndAddress(unsigned int segIndex) const
{
  return segActualLoadAddress(segIndex) + segSize(segIndex);
}

static uintptr_t read_uleb128(const uint8_t*& p, const uint8_t* end)
{
  uint64_t result = 0;
  int bit = 0;
  do {
    if (p == end)
      tvm::throwf_("malformed uleb128");

    uint64_t slice = *p & 0x7f;

    if (bit > 63)
      tvm::throwf_("uleb128 too big for uint64, bit=%d, result=0x%0llX", bit, result);
    else {
      result |= (slice << bit);
      bit += 7;
    }
  } while (*p++ & 0x80);
  return (uintptr_t)result;
}

const unsafe_loader::Symbol* unsafe_loader::findShallowExportedSymbol(const char* symbol) const
{
  //tvm::log_("Compressed::findExportedSymbol(%s) in %s\n", symbol, this->getShortName());
  uint32_t trieFileOffset = fDyldInfo ? fDyldInfo->export_off  : fExportsTrie->dataoff;
  uint32_t trieFileSize   = fDyldInfo ? fDyldInfo->export_size : fExportsTrie->datasize;
  if ( trieFileSize == 0 )
    return NULL;

  const uint8_t* start = &fLinkEditBase[trieFileOffset];
  const uint8_t* end = &start[trieFileSize];
  const uint8_t* foundNodeStart = this->trieWalk(start, end, symbol);
  if ( foundNodeStart != NULL ) {
    const uint8_t* p = foundNodeStart;
    const uintptr_t flags = read_uleb128(p, end);
    // found match, return pointer to terminal part of node
    if ( flags & EXPORT_SYMBOL_FLAGS_REEXPORT ) {
      // re-export from another dylib, lookup there
      const uintptr_t ordinal = read_uleb128(p, end);
      const char* importedName = (char*)p;
      if ( importedName[0] == '\0' )
        importedName = symbol;
      if (ordinal > 0) {
        tvm::throwf_("Symbol was found not in current module");
      }
    }
    else {
      return (Symbol*)foundNodeStart;
    }
  }
  return nullptr;
}




const uint8_t* unsafe_loader::trieWalk(const uint8_t* start, const uint8_t* end, const char* s) const
{
  const uint8_t* p = start;
  while ( p != NULL ) {
    uintptr_t terminalSize = *p++;
    if ( terminalSize > 127 ) {
      // except for re-export-with-rename, all terminal sizes fit in one byte
      --p;
      terminalSize = read_uleb128(p, end);
    }
    if ( (*s == '\0') && (terminalSize != 0) ) {
      //tvm::log_("trieWalk(%p) returning %p\n", start, p);
      return p;
    }
    const uint8_t* children = p + terminalSize;
    if ( children > end ) {
      tvm::log_("trieWalk() malformed trie node, terminalSize=0x%lx extends past end of trie\n", terminalSize);
      return NULL;
    }
    //tvm::log_("trieWalk(%p) sym=%s, terminalSize=%lu, children=%p\n", start, s, terminalSize, children);
    uint8_t childrenRemaining = *children++;
    p = children;
    uintptr_t nodeOffset = 0;
    for (; childrenRemaining > 0; --childrenRemaining) {
      const char* ss = s;
      //tvm::log_("trieWalk(%p) child str=%s\n", start, (char*)p);
      bool wrongEdge = false;
      // scan whole edge to get to next edge
      // if edge is longer than target symbol name, don't read past end of symbol name
      char c = *p;
      while ( c != '\0' ) {
        if ( !wrongEdge ) {
          if ( c != *ss )
            wrongEdge = true;
          ++ss;
        }
        ++p;
        c = *p;
      }
      if ( wrongEdge ) {
        // advance to next child
        ++p; // skip over zero terminator
        // skip over uleb128 until last byte is found
        while ( (*p & 0x80) != 0 )
          ++p;
        ++p; // skip over last byte of uleb128
        if ( p > end ) {
          tvm::log_("trieWalk() malformed trie node, child node extends past end of trie\n");
          return NULL;
        }
      }
      else {
        // the symbol so far matches this edge (child)
        // so advance to the child's node
        ++p;
        nodeOffset = read_uleb128(p, end);
        if ( (nodeOffset == 0) || ( &start[nodeOffset] > end) ) {
          tvm::log_("trieWalk() malformed trie child, nodeOffset=0x%lx out of range\n", nodeOffset);
          return NULL;
        }
        s = ss;
        //tvm::log_("trieWalk() found matching edge advancing to node 0x%lx\n", nodeOffset);
        break;
      }
    }
    if ( nodeOffset != 0 )
      p = &start[nodeOffset];
    else
      p = NULL;
  }
  //tvm::log_("trieWalk(%p) return NULL\n", start);
  return NULL;
}

uintptr_t unsafe_loader::exportedSymbolAddress(const LinkContext& context, const Symbol* symbol, bool runResolver) const
{
  uint32_t trieFileOffset = fDyldInfo ? fDyldInfo->export_off  : fExportsTrie->dataoff;
  uint32_t trieFileSize   = fDyldInfo ? fDyldInfo->export_size : fExportsTrie->datasize;
  const uint8_t* exportNode = (uint8_t*)symbol;
  const uint8_t* exportTrieStart = fLinkEditBase + trieFileOffset;
  const uint8_t* exportTrieEnd = exportTrieStart + trieFileSize;
  if ( (exportNode < exportTrieStart) || (exportNode > exportTrieEnd) )
    throw "symbol is not in trie";

  uintptr_t flags = read_uleb128(exportNode, exportTrieEnd);
  switch ( flags & EXPORT_SYMBOL_FLAGS_KIND_MASK ) {
    case EXPORT_SYMBOL_FLAGS_KIND_REGULAR:
      if ( runResolver && (flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER) ) {
        // this node has a stub and resolver, run the resolver to get target address
        uintptr_t stub = read_uleb128(exportNode, exportTrieEnd) + (uintptr_t)fMachOData; // skip over stub
        // <rdar://problem/10657737> interposing dylibs have the stub address as their replacee
        uintptr_t interposedStub = stub; //interposedAddress(context, stub, requestor);
        if ( interposedStub != stub )
          return interposedStub;
        // stub was not interposed, so run resolver
        typedef uintptr_t (*ResolverProc)(void);
        ResolverProc resolver = (ResolverProc)(read_uleb128(exportNode, exportTrieEnd) + (uintptr_t)fMachOData);
        uintptr_t result = (*resolver)();
        if ( context.verboseBind )
          tvm::log_("dyld: resolver at %p returned 0x%08lX\n", resolver, result);
        return result;
      }
      return read_uleb128(exportNode, exportTrieEnd) + (uintptr_t)fMachOData;
    case EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL:
      if ( flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER )
        tvm::throwf_("unsupported exported symbol kind. flags=%lu at node=%p", flags, symbol);
      return read_uleb128(exportNode, exportTrieEnd) + (uintptr_t)fMachOData;
    case EXPORT_SYMBOL_FLAGS_KIND_ABSOLUTE:
      if ( flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER )
        tvm::throwf_("unsupported exported symbol kind. flags=%lu at node=%p", flags, symbol);
      return read_uleb128(exportNode, exportTrieEnd);
    default:
      tvm::throwf_("unsupported exported symbol kind. flags=%lu at node=%p", flags, symbol);
  }
}

void* unsafe_loader::getExportSymbol(const char* sym_) const {
  LinkContext ctx;
  void * res = nullptr;
  auto sym = findShallowExportedSymbol(sym_);
  if ( sym != NULL )
    res = (void*)exportedSymbolAddress(ctx, sym, false);
  return res;
}

}