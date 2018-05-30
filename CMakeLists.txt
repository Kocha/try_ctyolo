cmake_minimum_required(VERSION 3.6)
project(try_ctyolo)

find_package( OpenMP )
if (OPENMP_FOUND) 
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(PkgConfig)
pkg_check_modules(OPENCV REQUIRED opencv)

add_definitions("-Wall -Iinc -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard  -Ofast ")
include_directories(inc)
include_directories(/home/pi/git/robidouille/raspicam_cv/)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(/home/pi/git/robidouille/raspicam_cv/)
link_directories(/home/pi/git/raspberrypi/userland/build/lib/)
link_directories(${OPENCV_LIBRARY_DIRS})
# SET(CMAKE_EXE_LINKER_FLAGS  "-pg")

set(SOURCE_FILES
        cqt_tyolo.c
        yad2k_yolo.c
        cqt_gen/cqt_debug.c
        cqt_gen/cqt_debug.h
        cqt_gen/cqt_gen.c
        cqt_gen/cqt_gen.h
        cqt_lib/BatchNormalization.c
        cqt_lib/Conv2d_same_1x1.c
        cqt_lib/Conv2d_same_3x3.c
        cqt_lib/cqt_lib.c
        cqt_lib/InputLayer.c
        cqt_lib/LeakyReLU.c
        cqt_lib/MaxPooling2D.c
        cqt_lib/numpy.c
        inc/cqt.h
        inc/cqt_keras.h
        inc/cqt_lib.h
        inc/cqt_net.h
        inc/cqt_type.h
        inc/numpy.h
        inc/SSE2NEON.h)

option(USE_DEBUG "Using debug option" OFF)
If(USE_DEBUG)
  add_definitions(-DDEBUG)
endif()

add_executable(try_ctyolo ${SOURCE_FILES})
target_link_libraries(try_ctyolo m)
target_link_libraries(try_ctyolo raspicamcv)
target_link_libraries(try_ctyolo mmal_core mmal mmal_util vcos bcm_host)
target_link_libraries(try_ctyolo ${OPENCV_LIBRARIES})

