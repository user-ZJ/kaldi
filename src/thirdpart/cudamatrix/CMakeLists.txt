cmake_minimum_required(VERSION 2.8)
project(kaldi-cudamatrix)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")


include_directories(
		${CMAKE_SOURCE_DIR}
        ${KALDI_ROOT_PATH}/tools/openfst/include
		/opt/intel/mkl/include
		 )

link_directories(
                /usr/local/lib
                /opt/intel/mkl/lib/intel64
                )

set(SOURCE_LIST 
		cu-array.cc
        cu-block-matrix.cc
        cu-common.cc
        cu-compressed-matrix.cc
        cu-math.cc
        cu-matrix.cc
        cu-packed-matrix.cc
        cu-rand.cc
        cu-sparse-matrix.cc
        cu-sp-matrix.cc
        cu-tp-matrix.cc
        cu-vector.cc
        cu-kernels.cu
		)


add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME}
                    mkl_intel_lp64 mkl_core mkl_sequential kaldi-base kaldi-matrix kaldi-util
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


