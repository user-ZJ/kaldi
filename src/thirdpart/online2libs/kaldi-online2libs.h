#ifndef KALDI_ONLINE2LIBS_H_
#define KALDI_ONLINE2LIBS_H_

#include "online2/online-ivector-feature.h"

using namespace kaldi;

int ivector_extract_online2(vec_ss* vss_spk2utt,map_ss* mss_feature,vec_ss* vss_ivectors,OnlineIvectorExtractionConfig& ivector_config,OnlineIvectorExtractionInfo& ivector_info);

#endif
