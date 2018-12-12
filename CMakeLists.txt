cmake_minimum_required( VERSION 2.8 )
project( Video_Stitching )

set( CMAKE_BUILD_TYPE "Debug" )
#set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set(EXECUTABLE_OUTPUT_PATH ../bin)
#find_package( OpenCV )
find_package( OpenCV 3.1 REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    include)
    
aux_source_directory(src DIR_SRCS)

add_executable( video_stitching ${DIR_SRCS})

target_link_libraries( video_stitching  ${OpenCV_LIBS} )
