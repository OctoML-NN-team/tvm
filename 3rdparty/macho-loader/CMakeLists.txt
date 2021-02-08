
set(TARGET macho_loader)

add_definitions(-DTVM_FOR=1)
#include_directories("include")

FILE(GLOB SRC src/*.cpp)
FILE(GLOB HDR src/*.h)

add_library(${TARGET} STATIC ${SRC} ${HDR})
target_include_directories(${TARGET} PUBLIC "include")