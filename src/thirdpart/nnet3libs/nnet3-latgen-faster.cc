// nnet3bin/nnet3-latgen-faster.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "base/timer.h"
#include "kaldi-nnet3libs.h"

int nnet3_latgen_faster(map_ss* mss_online_ivector,fst::SymbolTable *word_syms,TransitionModel &trans_model,AmNnetSimple &am_nnet,fst::Fst<fst::StdArc> *decode_fst,
                        vec_ss *vss_feature,vec_ss *vss_lattice,NnetSimpleComputationOptions &decodable_opts,LatticeFasterDecoderConfig &config) {
  // note: making this program work with GPUs is as simple as initializing the
  // device, but it probably won't make a huge difference in speed for typical
  // setups.  You should use nnet3-latgen-faster-batch if you want to use a GPU.
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    Timer timer;

    std::string ivector_rspecifier,
        utt2spk_rspecifier;

    std::string words_wspecifier="";
    std::string alignment_wspecifier="";

    bool allow_partial=true;
    int online_ivector_period=10;
    std::string range="";

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(vss_lattice)
           : lattice_writer.Open(vss_lattice)))
      KALDI_ERR << "Could not open table for writing lattices: ";

    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        mss_online_ivector,range);
    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
        ivector_rspecifier, utt2spk_rspecifier);

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    // this compiler object allows caching of computations across
    // different utterances.
    CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
                                       decodable_opts.optimize_config);

    if (decode_fst!=NULL) {
      SequentialBaseFloatMatrixReader feature_reader(vss_feature);

      // Input FST is just one FST, not a table of FSTs.
      //Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        LatticeFasterDecoder decoder(*decode_fst, config);

        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          const Matrix<BaseFloat> &features (feature_reader.Value());
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
          const Matrix<BaseFloat> *online_ivectors = NULL;
          const Vector<BaseFloat> *ivector = NULL;
          if (!ivector_rspecifier.empty()) {
            if (!ivector_reader.HasKey(utt)) {
              KALDI_WARN << "No iVector available for utterance " << utt;
              num_fail++;
              continue;
            } else {
              ivector = &ivector_reader.Value(utt);
            }
          }
          if (mss_online_ivector!=NULL) {
            if (!online_ivector_reader.HasKey(utt)) {
              KALDI_WARN << "No online iVector available for utterance " << utt;
              num_fail++;
              continue;
            } else {
              online_ivectors = &online_ivector_reader.Value(utt);
            }
          }

          DecodableAmNnetSimple nnet_decodable(
              decodable_opts, trans_model, am_nnet,
              features, ivector, online_ivectors,
              online_ivector_period, &compiler);

          double like;
          if (DecodeUtteranceLatticeFaster(
                  decoder, nnet_decodable, trans_model, word_syms, utt,
                  decodable_opts.acoustic_scale, determinize, allow_partial,
                  &alignment_writer, &words_writer, &compact_lattice_writer,
                  &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += nnet_decodable.NumFramesReady();
            num_success++;
          } else num_fail++;
        }
      }
      //delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      /*SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }

        LatticeFasterDecoder decoder(fst_reader.Value(), config);

        const Matrix<BaseFloat> *online_ivectors = NULL;
        const Vector<BaseFloat> *ivector = NULL;
        if (!ivector_rspecifier.empty()) {
          if (!ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No iVector available for utterance " << utt;
            num_fail++;
            continue;
          } else {
            ivector = &ivector_reader.Value(utt);
          }
        }
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No online iVector available for utterance " << utt;
            num_fail++;
            continue;
          } else {
            online_ivectors = &online_ivector_reader.Value(utt);
          }
        }

        DecodableAmNnetSimple nnet_decodable(
            decodable_opts, trans_model, am_nnet,
            features, ivector, online_ivectors,
            online_ivector_period, &compiler);

        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, nnet_decodable, trans_model, word_syms, utt,
                decodable_opts.acoustic_scale, determinize, allow_partial,
                &alignment_writer, &words_writer, &compact_lattice_writer,
                &lattice_writer, &like)) {
          tot_like += like;
          frame_count += nnet_decodable.NumFramesReady();
          num_success++;
        } else num_fail++;
      }*/
    }

    kaldi::int64 input_frame_count =
        frame_count * decodable_opts.frame_subsampling_factor;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed * 100.0 / input_frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over "
              << frame_count << " frames.";

    //delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
