cmake_minimum_required(VERSION 2.8)
project(KALDI_SS)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")
set(KALDI_ROOT_PATH "/home/zack/sourceCode/kaldi-modify")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

set(HAVE_CUDA TRUE)

add_definitions(-DHAVE_MKL)
add_definitions(-DKALDI_DOUBLEPRECISION=0)
if(HAVE_CUDA)
find_package(CUDA REQUIRED)
add_definitions(-DHAVE_CUDA)
endif()

add_subdirectory(kaldi)
#add_subdirectory(base)
#add_subdirectory(matrix)
#add_subdirectory(util)
#add_subdirectory(cudamatrix)
#add_subdirectory(tree)
#add_subdirectory(fstext)
#add_subdirectory(gmm)
#add_subdirectory(hmm)
#add_subdirectory(transform)
#add_subdirectory(ivector)
#add_subdirectory(lat)
#add_subdirectory(decoder)
#add_subdirectory(feat)
#add_subdirectory(online2)
#add_subdirectory(nnet3)
#add_subdirectory(featlibs)
#add_subdirectory(online2libs)
#add_subdirectory(nnet3libs)
#add_subdirectory(latlibs)
add_subdirectory(asrlib)
add_subdirectory(test)


