#ifndef ASR_H_
#define ASR_H_

#include<string>
#include<vector>
#include<sstream>
#include<iostream>
#include "feat/feature-mfcc.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"
#include "feat/kaldi-featlibs.h"
#if HAVE_CUDA == 1
#include "cudafeat/kaldi-cudafeatlibs.h"
#endif
#include "online2/kaldi-online2libs.h"
#include "nnet3/kaldi-nnet3libs.h"
#include "online2/online-ivector-feature.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "lat/kaldi-latlibs.h"

using namespace kaldi; 
using namespace kaldi::nnet3; 
using namespace fst;

namespace ASR{

class ASRRecog{
  public:
    ASRRecog();
    ~ASRRecog();

    int initialize(const std::string &config,int gpuid);
    int deInit();
    std::string predict(std::istream &wavStream,std::string fileId);
  private:
    std::string config;
    MfccOptions mfcc_opts;
    PitchExtractionOptions pitch_opts;
    ProcessPitchOptions process_opts;
    OnlineIvectorExtractionConfig ivector_opts;
    OnlineIvectorExtractionInfo *ivector_model;
    SymbolTable *word_syms;
    TransitionModel *trans_model;
    AmNnetSimple *am_nnet;
    Fst<StdArc> *decoder_fst;
    LatticeFasterDecoderConfig decoder_opts;
    NnetSimpleComputationOptions compute_opts;
};


}


#endif
