cmake_minimum_required(VERSION 2.8)
project(kaldi-latlibs)

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

set(SOURCE_LIST 
		lattice-scale.cc
        lattice-best-path.cc
		)

add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-matrix kaldi-util kaldi-tree kaldi-hmm kaldi-lat
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


