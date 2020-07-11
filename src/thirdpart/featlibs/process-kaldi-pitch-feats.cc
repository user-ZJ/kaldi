// featbin/process-kaldi-pitch-feats.cc

// Copyright 2013   Pegah Ghahremani
//                  Johns Hopkins University (author: Daniel Povey)
//           2014   IMSL, PKU-HKUST (author: Wei Shi)
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"
#include "kaldi-featlibs.h"

int process_kaldi_pitch_feats(vec_ss* vss_input,vec_ss* vss_output,ProcessPitchOptions& process_opts) {
  try {
    using namespace kaldi;

    int32 srand_seed = 0;
    srand(srand_seed);
    

    SequentialBaseFloatMatrixReader feat_reader(vss_input);
    BaseFloatMatrixWriter feat_writer(vss_output);

    int32 num_done = 0;
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &features = feat_reader.Value();

      Matrix<BaseFloat> processed_feats(features);
      ProcessPitch(process_opts, features, &processed_feats);

      feat_writer.Write(utt, processed_feats);
      num_done++;
    }
    KALDI_LOG << "Post-processed pitch for " << num_done << " utterances.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

