// featbin/apply-cmvn.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University

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
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"
#include "kaldi-featlibs.h"

int apply_cmvn(map_ss* mss_cmvn,vec_ss* vss_feature,vec_ss* vss_cmvn_output) {
  try {
    using namespace kaldi;

    bool norm_vars = false;
    bool norm_means = false;
    bool reverse = false;
    std::string skip_dims_str;
    std::string range=":,0:39";

    if (norm_vars && !norm_means)
      KALDI_ERR << "You cannot normalize the variance but not the mean.";

    if (!norm_means) {
      // CMVN is a no-op, we're not doing anything.  Just echo the input
      // don't even uncompress, if it was a CompressedMatrix.
      SequentialGeneralMatrixReader reader(vss_feature);
      GeneralMatrixWriter writer(vss_cmvn_output);
      kaldi::int32 num_done = 0;
      for (;!reader.Done(); reader.Next()) {
        writer.Write(reader.Key(), reader.Value());
        num_done++;
      }
      KALDI_LOG << "Copied " << num_done << " utterances.";
      return (num_done != 0 ? 0 : 1);
    }


    std::vector<int32> skip_dims;  // optionally use "fake"
                                   // (zero-mean/unit-variance) stats for some
                                   // dims to disable normalization.
    if (!SplitStringToIntegers(skip_dims_str, ":", false, &skip_dims)) {
      KALDI_ERR << "Bad --skip-dims option (should be colon-separated list of "
                <<  "integers)";
    }


    kaldi::int32 num_done = 0, num_err = 0;

    SequentialBaseFloatMatrixReader feat_reader(vss_feature);
    BaseFloatMatrixWriter feat_writer(vss_cmvn_output);

    if (mss_cmvn!=NULL){ // reading from a Table: per-speaker or per-utt CMN/CVN.
      //std::string cmvn_rspecifier = cmvn_rspecifier_or_rxfilename;
      //RandomAccessDoubleMatrixReaderMapped cmvn_reader(cmvn_rspecifier,utt2spk_rspecifier);
      RandomAccessDoubleMatrixReader cmvn_reader(mss_cmvn,range);

      for (; !feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> feat(feat_reader.Value());
        if (norm_means) {
          if (!cmvn_reader.HasKey(utt)) {
            KALDI_WARN << "No normalization statistics available for key "
                       << utt << ", producing no output for this utterance";
            num_err++;
            continue;
          }
          Matrix<double> cmvn_stats = cmvn_reader.Value(utt);
          if (!skip_dims.empty())
            FakeStatsForSomeDims(skip_dims, &cmvn_stats);

          if (reverse) {
            ApplyCmvnReverse(cmvn_stats, norm_vars, &feat);
          } else {
            ApplyCmvn(cmvn_stats, norm_vars, &feat);
          }
          feat_writer.Write(utt, feat);
        } else {
          feat_writer.Write(utt, feat);
        }
        num_done++;
      }
    } else {
      /*if (utt2spk_rspecifier != "")
        KALDI_ERR << "--utt2spk option not compatible with rxfilename as input "
                  << "(did you forget ark:?)";
      std::string cmvn_rxfilename = cmvn_rspecifier_or_rxfilename;
      bool binary;
      Input ki(cmvn_rxfilename, &binary);
      Matrix<double> cmvn_stats;
      cmvn_stats.Read(ki.Stream(), binary);
      if (!skip_dims.empty())
        FakeStatsForSomeDims(skip_dims, &cmvn_stats);

      for (;!feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> feat(feat_reader.Value());
        if (norm_means) {
          if (reverse) {
            ApplyCmvnReverse(cmvn_stats, norm_vars, &feat);
          } else {
            ApplyCmvn(cmvn_stats, norm_vars, &feat);
          }
        }
        feat_writer.Write(utt, feat);
        num_done++;
      }*/
    }
    if (norm_vars)
      KALDI_LOG << "Applied cepstral mean and variance normalization to "
                << num_done << " utterances, errors on " << num_err;
    else
      KALDI_LOG << "Applied cepstral mean normalization to "
                << num_done << " utterances, errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
