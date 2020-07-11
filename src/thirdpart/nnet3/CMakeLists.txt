cmake_minimum_required(VERSION 2.8)
project(kaldi-nnet3)

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

set(SOURCE_LIST 
		nnet-am-decodable-simple.cc
        nnet-optimize.cc
        nnet-compile.cc
        nnet-component-itf.cc
        nnet-common.cc
        nnet-parse.cc
        nnet-nnet.cc
        nnet-descriptor.cc
        nnet-computation.cc
        nnet-computation-graph.cc
        nnet-analyze.cc
        nnet-optimize-utils.cc
        nnet-compute.cc
        nnet-example.cc
        am-nnet-simple.cc
        nnet-utils.cc
        nnet-compile-utils.cc
        nnet-simple-component.cc
        natural-gradient-online.cc
        nnet-combined-component.cc
        nnet-normalize-component.cc
        nnet-general-component.cc
        nnet-convolutional-component.cc
        convolution.cc
        nnet-attention-component.cc
        attention.cc
        nnet-graph.cc
        nnet-diagnostics.cc
        nnet-example-utils.cc
        nnet-training.cc
        nnet-tdnn-component.cc
		)

add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME}
                    kaldi-base kaldi-matrix kaldi-util kaldi-cudamatrix kaldi-fstext kaldi-tree kaldi-hmm kaldi-lat kaldi-decoder fst
                )


INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


