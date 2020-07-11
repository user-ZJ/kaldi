cmake_minimum_required(VERSION 2.8)
project(kaldi-online2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")

include_directories(
		    ${CMAKE_SOURCE_DIR}
            ${KALDI_ROOT_PATH}/tools/openfst/include
		    )

link_directories(
                /usr/local/lib
                /opt/intel/mkl/lib/intel64
                ${KALDI_ROOT_PATH}/tools/openfst/lib
                )

set(SOURCE_LIST 
	    online-ivector-feature.cc
		)

add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-matrix kaldi-util kaldi-tree kaldi-gmm kaldi-hmm kaldi-ivector fst kaldi-decoder
                )

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


