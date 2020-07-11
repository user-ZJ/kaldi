#ifndef KALDI_LATLIBS_H_
#define KALDI_LATLIBS_H_


using namespace kaldi;

int lattice_scale(vec_ss* vss_lattice_input,vec_ss* vss_lattice_output);
int lattice_best_path(fst::SymbolTable *word_syms,vec_ss* vss_lats_input,vec_ss* vss_lats_output);

#endif
