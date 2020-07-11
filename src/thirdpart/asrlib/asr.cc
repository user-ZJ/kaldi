#include "asr.h"

using namespace ASR;

ASRRecog::ASRRecog() {}
ASRRecog::~ASRRecog() {}


void clearVSS(vec_ss *vss_input){
    for(vec_ss::iterator it=vss_input->begin();it!=vss_input->end();it++){
      delete it->getStrSS();
    }
    vss_input->clear();
}

void clearMSS(map_ss *map_input){
    for(map_ss::iterator it=map_input->begin();it!=map_input->end();it++){
        delete it->second;
        it->second=NULL;
        map_input->erase(it);
    }
}

int ASRRecog::initialize(const std::string &config,int gpu_id){
  mfcc_opts.frame_opts.samp_freq = 16000;
  mfcc_opts.mel_opts.num_bins = 40;
  mfcc_opts.use_energy = false;
  mfcc_opts.num_ceps = 40;
  mfcc_opts.mel_opts.low_freq = 40;
  mfcc_opts.mel_opts.high_freq = -200;

  pitch_opts.samp_freq = 16000;

  ivector_opts.cmvn_config_rxfilename = "/home/zack/sourceCode/kaldi-master/src/thirdpart/configure/online_cmvn.conf";
  ivector_opts.ivector_period =10;
  ivector_opts.splice_config_rxfilename="/home/zack/sourceCode/kaldi-master/src/thirdpart/configure/splice.conf";
  ivector_opts.lda_mat_rxfilename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/final.mat";
  ivector_opts.global_cmvn_stats_rxfilename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/global_cmvn.stats";
  ivector_opts.diag_ubm_rxfilename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/final.dubm";
  ivector_opts.ivector_extractor_rxfilename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/final.ie";
  ivector_opts.num_gselect=5;
  ivector_opts.min_post=0.025;
  ivector_opts.posterior_scale=0.1;
  ivector_opts.max_remembered_frames=1000;
  ivector_opts.max_count=0;
  ivector_opts.use_most_recent_ivector=false;

  ivector_model=new OnlineIvectorExtractionInfo(ivector_opts);


  decoder_opts.max_active = 7000;
  decoder_opts.min_active = 200;
  decoder_opts.beam = 15.0;
  decoder_opts.lattice_beam = 8.0;

  compute_opts.frame_subsampling_factor = 3;
  compute_opts.frames_per_chunk = 50;
  compute_opts.extra_left_context = 0;
  compute_opts.extra_right_context = 0;
  compute_opts.extra_left_context_initial = -1;
  compute_opts.extra_right_context_final = -1;
  compute_opts.acoustic_scale = 1.0;

  std::string word_syms_filename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/words.txt";
  std::string fst_filename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/HCLG.fst";
  std::string asr_model_filename="/home/zack/sourceCode/kaldi-master/src/thirdpart/model/asr/final.mdl";
  word_syms = fst::SymbolTable::ReadText(word_syms_filename);
  if(word_syms==NULL) KALDI_ERR<<"read word syms "<<word_syms_filename<<" failed";
  decoder_fst=fst::ReadFstKaldiGeneric(fst_filename);
  if(decoder_fst==NULL) KALDI_ERR<<"read decoder_fst "<<fst_filename<<" failed";
  bool binary;
  Input ki(asr_model_filename,&binary);
  trans_model = new TransitionModel();
  am_nnet = new AmNnetSimple();
  trans_model->Read(ki.Stream(),binary);
  am_nnet->Read(ki.Stream(),binary);
  SetBatchnormTestMode(true,&(am_nnet->GetNnet()));
  SetDropoutTestMode(true,&(am_nnet->GetNnet()));
  CollapseModel(CollapseModelConfig(),&(am_nnet->GetNnet()));

}

int ASRRecog::deInit(){
  return 1;
}

std::string ASRRecog::predict(std::istream &wav_stream,std::string file_id){
  KALDI_LOG<<"asr predict start";
  if(wav_stream.fail()){
    KALDI_WARN<<"wav file not exist!";
    return "Error:wav file not exist";
  }
  int ret = 0;
  //read wav data
  WaveHolder holder = WaveHolder();
  if(!holder.Read(wav_stream)){
    KALDI_WARN<<"read wav file error";
    return "Error:read wav file error";
  }
  WaveData &wave_data = holder.Value();
  int sample_rate = wave_data.SampFreq();
  KALDI_LOG<<"wav sample rate is:"<<sample_rate;
  
  vec_ss vss_mfcc_output,vss_cpitch_output,vss_ppitch_output,vss_paste_feats_output,vss_copy_feats_output,
         vss_cmvn_spk2utt_input,vss_cmvn_stats_output,vss_spk2utt_ivector_input,vss_ivector_output,vss_apply_cmvn_output,
         vss_latgen_output,vss_lattice_scale_output,vss_lattice_best_output;
  vec_ss_list vss_input_list;
  map_ss_list map_input_list;
  map_ss map_pro_pitch,map_latgen_input,map_apply_cmvn_input,map_cmvn_stats_input,map_ivector_input;

  vss_mfcc_output.push_back(StrSS(file_id,new std::stringstream));
  ret = compute_mfcc_feats(wave_data,file_id,&vss_mfcc_output,mfcc_opts);
  if(ret){
    KALDI_ERR<<"compute mfcc failed";
    return "Error:compute mfcc failed";
  }
  vss_cpitch_output.push_back(StrSS(file_id,new std::stringstream));
  ret = compute_kaldi_pitch_feats(wave_data,file_id,&vss_cpitch_output,pitch_opts);
  if(ret){
    KALDI_WARN<<"compute pitch failed";
    return "Error:compute pitch failed";
  }
  vss_ppitch_output.push_back(StrSS(file_id,new std::stringstream));
  ret = process_kaldi_pitch_feats(&vss_cpitch_output,&vss_ppitch_output,process_opts);
  if(ret){
    KALDI_WARN<<"process pitch failed";
    return "Error:process pitch failed";
  }
  vss_input_list.push_back(vss_mfcc_output);
  for(vec_ss::iterator it=vss_ppitch_output.begin();it!=vss_ppitch_output.end();it++){
    map_pro_pitch.insert(std::pair<std::string,std::stringstream*>(it->getString(),it->getStrSS()));
  }
  map_input_list.push_back(map_pro_pitch);
  vss_paste_feats_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = paste_feats(&vss_input_list,&map_input_list,&vss_paste_feats_output);
  if(ret){
    KALDI_ERR<<"paste feats failed";
    return "Error:paste feats failed";
  }
  vss_copy_feats_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = copy_feats(&vss_paste_feats_output,&vss_copy_feats_output);
  if(ret){
    KALDI_ERR<<"copy feats failed";
    return "Error:copy feats failed";
  }
  vss_cmvn_spk2utt_input.push_back(StrSS(file_id,new std::stringstream(file_id)));
  for(vec_ss::iterator it=vss_copy_feats_output.begin();it!=vss_copy_feats_output.end();it++){
    map_cmvn_stats_input.insert(std::pair<std::string,std::stringstream*>(it->getString(),new std::stringstream(it->getStrSS()->str())));
    map_ivector_input.insert(std::pair<std::string,std::stringstream*>(it->getString(),new std::stringstream(it->getStrSS()->str())));
  }
  vss_cmvn_stats_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = compute_cmvn_stats(&vss_cmvn_spk2utt_input,&map_cmvn_stats_input,&vss_cmvn_stats_output);
  if(ret){
    KALDI_ERR<<"compute cmvn failed";
    return "Error:compute cmvn failed";
  }
  vss_spk2utt_ivector_input.push_back(StrSS(file_id,new std::stringstream(file_id)));
  vss_ivector_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = ivector_extract_online2(&vss_spk2utt_ivector_input,&map_ivector_input,&vss_ivector_output,ivector_opts,*ivector_model);
  if(ret){
      KALDI_ERR<<"ivector extract failed";
      return "Error:ivector extract failed";
  }
  for(vec_ss::iterator it=vss_cmvn_stats_output.begin();it!=vss_cmvn_stats_output.end();it++){
      map_apply_cmvn_input.insert(std::pair<std::string,std::stringstream*>(it->getString(),it->getStrSS()));
  }
  vss_apply_cmvn_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = apply_cmvn(&map_apply_cmvn_input,&vss_copy_feats_output,&vss_apply_cmvn_output);
  if(ret){
      KALDI_ERR<<"apply cmvn failed";
      return "Error:apply cmvn failed";
  }
  for(vec_ss::iterator it=vss_ivector_output.begin();it!=vss_ivector_output.end();it++){
      map_latgen_input.insert(std::pair<std::string,std::stringstream*>(it->getString(),it->getStrSS()));
  }
  vss_latgen_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = nnet3_latgen_faster(&map_latgen_input,word_syms,*trans_model,*am_nnet,decoder_fst,&vss_apply_cmvn_output,&vss_latgen_output,compute_opts,decoder_opts);
  if(ret){
      KALDI_ERR<<"latgen failed";
      return "Error:latgen failed";
  }
  vss_lattice_scale_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = lattice_scale(&vss_latgen_output,&vss_lattice_scale_output);
  if(ret){
      KALDI_ERR<<"lattice scale failed";
      return "Error:lattice scale failed";
  }
  vss_lattice_best_output.push_back(StrSS(file_id,new std::stringstream()));
  ret = lattice_best_path(word_syms,&vss_lattice_scale_output,&vss_lattice_best_output);
  if(ret){
      KALDI_ERR<<"lattic best path failed";
      return "Error:lattic best path failed";
  }
  clearVSS(&vss_cmvn_spk2utt_input);
  clearVSS(&vss_spk2utt_ivector_input);
  clearVSS(&vss_mfcc_output);
  clearVSS(&vss_cpitch_output);
  clearVSS(&vss_ppitch_output);
  clearVSS(&vss_paste_feats_output);
  clearVSS(&vss_copy_feats_output);
  clearVSS(&vss_cmvn_stats_output);
  clearVSS(&vss_ivector_output);
  clearVSS(&vss_apply_cmvn_output);
  clearVSS(&vss_latgen_output);
  clearVSS(&vss_lattice_best_output);
  clearVSS(&vss_lattice_scale_output);

  clearMSS(&map_cmvn_stats_input);
  clearMSS(&map_ivector_input);

  vss_input_list.clear();
  map_input_list.clear();

  return "test";
}
