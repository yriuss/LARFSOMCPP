cmake_minimum_required (VERSION 3.14.0)
project (BOW_opencv C CXX)
find_package (GeographicLib REQUIRED)
find_package(Matplot++ REQUIRED)


include(CheckLanguage)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
   endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(
${PROJECT_SOURCE_DIR}
${GeographicLib_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
src/LARFSOM.cc
include/LARFSOM.h
)

target_link_libraries (${PROJECT_NAME}
${MATPLOT_LIBRARIES}
${OpenCV_LIBS}
${GeographicLib_LIBRARIES}
)


# examples of this library
add_executable (color_segmentation ../examples/color_segmentation.cc)
target_link_libraries(color_segmentation ${PROJECT_NAME})