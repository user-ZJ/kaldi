cmake_minimum_required(VERSION 2.8)
project(asr)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories(
		${CMAKE_SOURCE_DIR}
                ${KALDI_ROOT_PATH}/tools/openfst/include
		 )

link_directories(
                /usr/local/lib
                ${KALDI_ROOT_PATH}/tools/openfst/lib
                /opt/intel/mkl/lib/intel64
                )


set(FEATBINS_SOURCES 
		asr.cc
		)

add_library(${PROJECT_NAME} SHARED ${FEATBINS_SOURCES})

target_link_libraries( ${PROJECT_NAME}
                    fst mkl_intel_lp64 mkl_core mkl_sequential kaldi-base kaldi-util kaldi-matrix kaldi-feat kaldi-nnet3 kaldi-featlibs kaldi-online2libs kaldi-nnet3libs kaldi-latlibs 
                )

#fst mkl_intel_lp64 mkl_core mkl_sequential kaldi-base kaldi-util kaldi-feat kaldi-featlibs kaldi-matrix

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


