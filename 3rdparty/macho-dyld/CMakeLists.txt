
set(TARGET macho_dyld)

add_definitions(-DTVM_FOR=1)

FILE(GLOB SRC src/*.cpp src/dyld/*.cpp)
FILE(GLOB HDR include/*.h src/*.h src/dyld/*.h)

add_library(${TARGET} STATIC ${SRC} ${HDR})
target_include_directories(${TARGET} PUBLIC "include")