// latbin/lattice-best-path.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "kaldi-latlibs.h"

int lattice_best_path(fst::SymbolTable *word_syms,vec_ss* vss_lats_input,vec_ss* vss_lats_output) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_scale = 15.0;

    std::string alignments_wspecifier = "";

    SequentialCompactLatticeReader clat_reader(vss_lats_input);

    Int32VectorWriter transcriptions_writer(vss_lats_output);

    Int32VectorWriter alignments_writer(alignments_wspecifier);

    int32 n_done = 0, n_fail = 0;
    int64 n_frame = 0;
    LatticeWeight tot_weight = LatticeWeight::One();

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
      CompactLattice clat_best_path;
      CompactLatticeShortestPath(clat, &clat_best_path);  // A specialized
      // implementation of shortest-path for CompactLattice.
      Lattice best_path;
      ConvertLattice(clat_best_path, &best_path);
      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Best-path failed for key " << key;
        n_fail++;
      } else {
        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
        KALDI_LOG << "For utterance " << key << ", best cost "
                  << weight.Value1() << " + " << weight.Value2() << " = "
                  << (weight.Value1() + weight.Value2())
                  << " over " << alignment.size() << " frames.";
        if (vss_lats_output != NULL)
          transcriptions_writer.Write(key, words);
        if (alignments_wspecifier != "")
          alignments_writer.Write(key, alignment);
        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        n_done++;
        n_frame += alignment.size();
        tot_weight = Times(tot_weight, weight);
      }
    }

    BaseFloat tot_weight_float = tot_weight.Value1() + tot_weight.Value2();
    KALDI_LOG << "Overall cost per frame is " << (tot_weight_float/n_frame)
              << " = " << (tot_weight.Value1()/n_frame) << " [graph]"
              << " + " << (tot_weight.Value2()/n_frame) << " [acoustic]"
              << " over " << n_frame << " frames.";
    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;

    //delete word_syms;
    if (n_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
