// latbin/lattice-scale.cc

// Copyright 2009-2013  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "kaldi-latlibs.h"

int lattice_scale(vec_ss* vss_lattice_input,vec_ss* vss_lattice_output) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    bool write_compact = true;
    BaseFloat acoustic_scale = 10.0;
    BaseFloat inv_acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    BaseFloat acoustic2lm_scale = 0.0;
    BaseFloat lm2acoustic_scale = 0.0;

    int32 n_done = 0;

    KALDI_ASSERT(acoustic_scale == 1.0 || inv_acoustic_scale == 1.0);
    if (inv_acoustic_scale != 1.0)
      acoustic_scale = 1.0 / inv_acoustic_scale;

    std::vector<std::vector<double> > scale(2);
    scale[0].resize(2);
    scale[1].resize(2);
    scale[0][0] = lm_scale;
    scale[0][1] = acoustic2lm_scale;
    scale[1][0] = lm2acoustic_scale;
    scale[1][1] = acoustic_scale;

    if (write_compact) {
      SequentialCompactLatticeReader compact_lattice_reader(vss_lattice_input);

      // Write as compact lattice.
      CompactLatticeWriter compact_lattice_writer(vss_lattice_output);

      for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
        CompactLattice lat = compact_lattice_reader.Value();
        ScaleLattice(scale, &lat);
        compact_lattice_writer.Write(compact_lattice_reader.Key(), lat);
        n_done++;
      }
    } else {
      SequentialLatticeReader lattice_reader(vss_lattice_input);

      // Write as regular lattice.
      LatticeWriter lattice_writer(vss_lattice_output);

      for (; !lattice_reader.Done(); lattice_reader.Next()) {
        Lattice lat = lattice_reader.Value();
        ScaleLattice(scale, &lat);
        lattice_writer.Write(lattice_reader.Key(), lat);
        n_done++;
      }
    }

    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
