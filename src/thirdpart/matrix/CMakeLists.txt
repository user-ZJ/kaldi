cmake_minimum_required(VERSION 2.8)
project(kaldi-matrix)

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
		compressed-matrix.cc
		kaldi-matrix.cc
		kaldi-vector.cc
		matrix-functions.cc
		optimization.cc
		packed-matrix.cc
		sp-matrix.cc
		sparse-matrix.cc
		srfft.cc
		tp-matrix.cc
                qr.cc
                srfft.cc
		)


add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME}
                    mkl_intel_lp64 mkl_core mkl_sequential kaldi-base
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


