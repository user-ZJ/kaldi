// online2bin/ivector-extract-online2.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "online2/online-ivector-feature.h"
#include "util/kaldi-thread.h"
#include "base/timer.h"
#include "kaldi-online2libs.h"

int ivector_extract_online2(vec_ss* vss_spk2utt,map_ss* mss_feature,vec_ss* vss_ivectors,OnlineIvectorExtractionConfig& ivector_config,OnlineIvectorExtractionInfo& ivector_info) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {

    g_num_threads = 8;
    bool repeat = false;
    int32 length_tolerance = 0;
    std::string frame_weights_rspecifier;

    double tot_ubm_loglike = 0.0, tot_objf_impr = 0.0, tot_t = 0.0,
        tot_length = 0.0, tot_length_utt_end = 0.0;
    int32 num_done = 0, num_err = 0;

    std::string range = ":,0:39";
    SequentialTokenVectorReader spk2utt_reader(vss_spk2utt);
    RandomAccessBaseFloatMatrixReader feature_reader(mss_feature,range);
    RandomAccessBaseFloatVectorReader frame_weights_reader(frame_weights_rspecifier);
    BaseFloatMatrixWriter ivector_writer(vss_ivectors);

    bool warned_dim = false;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      OnlineIvectorExtractorAdaptationState adaptation_state(
          ivector_info);
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value(utt);

        int32 feat_dim = feats.NumCols();
        if (feat_dim == ivector_info.ExpectedFeatureDim() + 3) {
          if (!warned_dim) {
            KALDI_WARN << "Feature dimension is too large by 3, assuming there are "
                "pitch features and removing the last 3 dims.";
            warned_dim = true;
          }
          feat_dim -= 3;
        }

        SubMatrix<BaseFloat> range = feats.ColRange(0, feat_dim);
        OnlineMatrixFeature matrix_feature(range);

        OnlineIvectorFeature ivector_feature(ivector_info,
                                             &matrix_feature);

        ivector_feature.SetAdaptationState(adaptation_state);

        if (!frame_weights_rspecifier.empty()) {
          if (!frame_weights_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find weights for utterance " << utt;
            num_err++;
            continue;
          }
          const Vector<BaseFloat> &weights = frame_weights_reader.Value(utt);

          if (std::abs(weights.Dim() - feats.NumRows()) > length_tolerance) {
            num_err++;
            continue;
          }

          std::vector<std::pair<int32, BaseFloat> > frame_weights;
          for (int32 i = 0; i < feats.NumRows(); i++) {
            if (i < weights.Dim())
              frame_weights.push_back(std::make_pair(i, weights(i)));
            else
              frame_weights.push_back(std::make_pair(i, 0.0));
          }


          ivector_feature.UpdateFrameWeights(frame_weights);
        }

        int32 T = feats.NumRows(),
            n = (repeat ? 1 : ivector_config.ivector_period),
            num_ivectors = (T + n - 1) / n;

        Matrix<BaseFloat> ivectors(num_ivectors,
                                   ivector_feature.Dim());

        for (int32 i = 0; i < num_ivectors; i++) {
          int32 t = i * n;
          SubVector<BaseFloat> ivector(ivectors, i);
          ivector_feature.GetFrame(t, &ivector);
        }
        // Update diagnostics.

        tot_ubm_loglike += T * ivector_feature.UbmLogLikePerFrame();
        tot_objf_impr += T * ivector_feature.ObjfImprPerFrame();
        tot_length_utt_end += T * ivectors.Row(num_ivectors - 1).Norm(2.0);
        for (int32 i = 0; i < num_ivectors; i++)
          tot_length += T * ivectors.Row(i).Norm(2.0) / num_ivectors;
        tot_t += T;
        KALDI_VLOG(2) << "For utterance " << utt << " of speaker " << spk
                      << ", UBM loglike/frame was "
                      << ivector_feature.UbmLogLikePerFrame()
                      << ", iVector length (at utterance end) was "
                      << ivectors.Row(num_ivectors-1).Norm(2.0)
                      << ", objf improvement/frame from iVector estimation was "
                      << ivector_feature.ObjfImprPerFrame();

        ivector_feature.GetAdaptationState(&adaptation_state);
        ivector_writer.Write(utt, ivectors);
        num_done++;
      }
    }

    KALDI_LOG << "Estimated iVectors for " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Average objective-function improvement was "
              << (tot_objf_impr / tot_t) << " per frame, over "
              << tot_t << " frames (weighted).";
    KALDI_LOG << "Average iVector length was " << (tot_length / tot_t)
              << " and at utterance-end was " << (tot_length_utt_end / tot_t)
              << ", over " << tot_t << " frames (weighted); "
              << " expected length is "
              << sqrt(ivector_info.extractor.IvectorDim());

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
