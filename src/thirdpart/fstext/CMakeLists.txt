cmake_minimum_required(VERSION 2.8)
project(kaldi-fstext)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories(
		${CMAKE_SOURCE_DIR}
                ${KALDI_ROOT_PATH}/tools/openfst/include
		 )

link_directories(
                 /usr/local/lib
                 /opt/intel/mkl/lib/intel64
                 ${KALDI_ROOT_PATH}/tools/openfst/lib
                 )

set(KALDI_BASE_SOURCES 
		context-fst.cc
        kaldi-fst-io.cc
        grammar-context-fst.cc
		)

add_library(${PROJECT_NAME} SHARED ${KALDI_BASE_SOURCES})

target_link_libraries(${PROJECT_NAME}
                     kaldi-base kaldi-matrix kaldi-util fst
                 )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


