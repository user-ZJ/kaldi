cmake_minimum_required(VERSION 2.8)
project(kaldi-ivector)

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
	    ivector-extractor.cc
		)

add_library(${PROJECT_NAME} SHARED ${KALDI_FEAT_SOURCES})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-matrix kaldi-util kaldi-gmm
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


