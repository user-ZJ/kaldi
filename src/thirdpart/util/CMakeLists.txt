cmake_minimum_required(VERSION 2.8)
project(kaldi-util)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories(
		${CMAKE_SOURCE_DIR}
                ${KALDI_ROOT_PATH}/tools/openfst/include
		 )

link_directories(
                /usr/local/lib
                /opt/intel/mkl/lib/intel64
                )

set(SOURCE_LIST 
		kaldi-holder.cc
		kaldi-io.cc
		kaldi-semaphore.cc
		kaldi-table.cc
		parse-options.cc
		simple-io-funcs.cc
		text-utils.cc
        kaldi-thread.cc
		)

add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-matrix
                )


INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


