// featbin/compute-kaldi-pitch-feats.cc

// Copyright 2013        Pegah Ghahremani
//           2013-2014   Johns Hopkins University (author: Daniel Povey)
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

int compute_kaldi_pitch_feats(const WaveData &wave_data,std::string file_id,vec_ss* vss_output,PitchExtractionOptions& pitch_opts) {
  try {
    using namespace kaldi;

    int32 channel = -1; // Note: this isn't configurable because it's not a very
                        // good idea to control it this way: better to extract the
                        // on the command line (in the .scp file) using sox or
                        // similar.

    BaseFloatMatrixWriter feat_writer(vss_output);
    //BaseFloatMatrixWriter feat_writer;
    //feat_writer.Open(vss_output);

    int32 num_done = 0, num_err = 0;
    //for (; !wav_reader.Done(); wav_reader.Next()) {
    for (int i=0;i<1;i++) {
      std::string utt = file_id;

      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {
        KALDI_ASSERT(num_chan > 0);
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }

      if (pitch_opts.samp_freq != wave_data.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << pitch_opts.samp_freq << " but data has "
                  << wave_data.SampFreq() << " (use --sample-frequency "
                  << "option).  Utterance is " << utt;


      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        ComputeKaldiPitch(pitch_opts, waveform, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute pitch for utterance "
                   << utt;
        num_err++;
        continue;
      }

      feat_writer.Write(utt, features);
      if (num_done % 50 == 0 && num_done != 0)
        KALDI_VLOG(2) << "Processed " << num_done << " utterances";
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

