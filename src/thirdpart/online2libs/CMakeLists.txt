cmake_minimum_required(VERSION 2.8)
project(kaldi-online2libs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")

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
		ivector-extract-online2.cc
		)

add_library(${PROJECT_NAME} SHARED ${FEATBINS_SOURCES})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-util kaldi-feat kaldi-online2
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


