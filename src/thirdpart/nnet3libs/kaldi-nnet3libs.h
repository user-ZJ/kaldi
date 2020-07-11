#ifndef KALDI_NNET3LIBS_H_
#define KALDI_NNET3LIBS_H_

#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"

using namespace kaldi;
using namespace kaldi::nnet3;

int nnet3_latgen_faster(map_ss* mss_online_ivector,fst::SymbolTable *word_syms,TransitionModel &trans_model,AmNnetSimple &am_nnet,fst::Fst<fst::StdArc> *decode_fst,
                         vec_ss *vss_feature,vec_ss *vss_lattice,NnetSimpleComputationOptions &decodable_opts,LatticeFasterDecoderConfig &config);

#endif
