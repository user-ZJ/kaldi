#ifndef KALDI_FEATLIBS_H_
#define KALDI_FEATLIBS_H_

#include "feat/pitch-functions.h"
#include "feat/feature-functions.h"
#include "feat/wave-reader.h"
#include "feat/feature-mfcc.h"

using namespace kaldi;

int compute_mfcc_feats(const WaveData &wave_data,std::string file_id,vec_ss* output,MfccOptions &mfcc_opts);
int compute_kaldi_pitch_feats(const WaveData &wave_data,std::string file_id,vec_ss* vss_output,PitchExtractionOptions &pitch_opts);
int process_kaldi_pitch_feats(vec_ss* vss_input,vec_ss* vss_output,ProcessPitchOptions& process_opts);
int paste_feats(vec_ss_list* vec_ss_input,map_ss_list* map_ss_input,vec_ss* vss_output);
int compute_cmvn_stats(vec_ss* vss_spk2utt,map_ss* map_mfcc,vec_ss* vss_output);
int copy_feats(vec_ss* vss_input,vec_ss* vss_output);
int apply_cmvn(map_ss* mss_cmvn,vec_ss* vss_feature,vec_ss* vss_cmvn_output);

#endif
