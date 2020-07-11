// featbin/copy-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "kaldi-featlibs.h"

int copy_feats(vec_ss* vss_input,vec_ss* vss_output) {
  try {
    using namespace kaldi;


    bool binary = true;
    bool htk_in = false;
    bool sphinx_in = false;
    bool compress = false;
    int32 compression_method_in = 1;
    std::string num_frames_wspecifier;

    int32 num_done = 0;

    CompressionMethod compression_method = static_cast<CompressionMethod>(
        compression_method_in);

    if (vss_input!=NULL) {
      // Copying tables of features.
      Int32Writer num_frames_writer(num_frames_wspecifier);

      if (!compress) {
        BaseFloatMatrixWriter kaldi_writer(vss_output);
        if (htk_in) {
          /*SequentialTableReader<HtkMatrixHolder> htk_reader(rspecifier);
          for (; !htk_reader.Done(); htk_reader.Next(), num_done++) {
            kaldi_writer.Write(htk_reader.Key(), htk_reader.Value().first);
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(htk_reader.Key(),
                                      htk_reader.Value().first.NumRows());
          }*/
        } else if (sphinx_in) {
          /*SequentialTableReader<SphinxMatrixHolder<> > sphinx_reader(rspecifier);
          for (; !sphinx_reader.Done(); sphinx_reader.Next(), num_done++) {
            kaldi_writer.Write(sphinx_reader.Key(), sphinx_reader.Value());
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(sphinx_reader.Key(),
                                      sphinx_reader.Value().NumRows());
          }*/
        } else {
          SequentialBaseFloatMatrixReader kaldi_reader(vss_input);
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
            kaldi_writer.Write(kaldi_reader.Key(), kaldi_reader.Value());
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
        }
      } else {
        CompressedMatrixWriter kaldi_writer(vss_output);
        if (htk_in) {
          /*SequentialTableReader<HtkMatrixHolder> htk_reader(rspecifier);
          for (; !htk_reader.Done(); htk_reader.Next(), num_done++) {
            kaldi_writer.Write(htk_reader.Key(),
                               CompressedMatrix(htk_reader.Value().first,
                                                compression_method));
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(htk_reader.Key(),
                                      htk_reader.Value().first.NumRows());
          }*/
        } else if (sphinx_in) {
          /*SequentialTableReader<SphinxMatrixHolder<> > sphinx_reader(rspecifier);
          for (; !sphinx_reader.Done(); sphinx_reader.Next(), num_done++) {
            kaldi_writer.Write(sphinx_reader.Key(),
                               CompressedMatrix(sphinx_reader.Value(),
                                                compression_method));
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(sphinx_reader.Key(),
                                      sphinx_reader.Value().NumRows());
          }*/
        } else {
          SequentialBaseFloatMatrixReader kaldi_reader(vss_input);
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
            kaldi_writer.Write(kaldi_reader.Key(),
                               CompressedMatrix(kaldi_reader.Value(),
                                                compression_method));
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
        }
      }
      KALDI_LOG << "Copied " << num_done << " feature matrices.";
      return (num_done != 0 ? 0 : 1);
    } else {
      /*KALDI_ASSERT(!compress && "Compression not yet supported for single files");
      if (!num_frames_wspecifier.empty())
        KALDI_ERR << "--write-num-frames option not supported when writing/reading "
                  << "single files.";

      std::string feat_rxfilename = po.GetArg(1), feat_wxfilename = po.GetArg(2);

      Matrix<BaseFloat> feat_matrix;
      if (htk_in) {
        Input ki(feat_rxfilename); // Doesn't look for read binary header \0B, because
        // no bool* pointer supplied.
        HtkHeader header; // we discard this info.
        ReadHtk(ki.Stream(), &feat_matrix, &header);
      } else if (sphinx_in) {
        KALDI_ERR << "For single files, sphinx input is not yet supported.";
      } else {
        ReadKaldiObject(feat_rxfilename, &feat_matrix);
      }
      WriteKaldiObject(feat_matrix, feat_wxfilename, binary);
      KALDI_LOG << "Copied features from " << PrintableRxfilename(feat_rxfilename)
                << " to " << PrintableWxfilename(feat_wxfilename);*/
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
