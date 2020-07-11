cmake_minimum_required(VERSION 2.8)
project(kaldi-base)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories(
		${CMAKE_SOURCE_DIR}
                ${KALDI_ROOT_PATH}/tools/openfst/include
		 )

set(SOURCE_LIST 
		io-funcs.cc
		kaldi-error.cc
		kaldi-math.cc
		kaldi-utils.cc
		timer.cc
		)

add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


