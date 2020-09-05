cmake_minimum_required(VERSION 2.8)
project(asr_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")

include_directories(
		${CMAKE_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/kaldi
        ${KALDI_ROOT_PATH}/tools/openfst/include
        /usr/local/cuda/include
		 )
link_directories(
		/usr/local/lib
        ${KALDI_ROOT_PATH}/tools/openfst/lib
        /opt/intel/mkl/lib/intel64
        /opt/intel/lib/intel64
        /usr/local/cuda/lib64
		)


set(FEATBINS_SOURCES 
		asr_test.cc
		)

add_executable(${PROJECT_NAME} ${FEATBINS_SOURCES})

target_link_libraries(${PROJECT_NAME}
		kaldiasr asr
		)
#fst mkl_intel_lp64 mkl_core mkl_sequential kaldi-base kaldi-util kaldi-feat asr kaldi-featlibs kaldi-matrix
INSTALL(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)


