cmake_minimum_required(VERSION 2.8)
project(kaldi-tree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories(
		${CMAKE_SOURCE_DIR}
                ${KALDI_ROOT_PATH}/tools/openfst/include
		 )

link_directories(
                /usr/local/lib
                /opt/intel/mkl/lib/intel64
                )

set(KALDI_FEAT_SOURCES 
	    cluster-utils.cc
        clusterable-classes.cc
        context-dep.cc
        event-map.cc
        build-tree.cc
        build-tree-utils.cc
        build-tree-questions.cc
		)

add_library(${PROJECT_NAME} SHARED ${KALDI_FEAT_SOURCES})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-matrix 
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


