#include<iostream>
#include "asrlib/asr.h"

using namespace ASR;
using namespace std;

void callASR(ASRRecog &asrRecog,string wavPath){
  int N = 1;
  for(int i=0;i<N;i++){
    fstream data_input;
    data_input.open(wavPath,ios::binary|ios::in);
    std::string result = asrRecog.predict(data_input,"ASR_WAV");
    data_input.close();
    cout<<result<<endl;
  }
  
}

int main(int argc,char *argv[]){
  string wavPath = argv[1];
  cout<<"wavPath:"<<wavPath<<endl;
  ASRRecog asrRecog;
  int ret = asrRecog.initialize("",0);
  callASR(asrRecog,wavPath);
}
