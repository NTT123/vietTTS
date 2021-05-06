if [ ! -f assets/reinfo/waveRNN/checkpoint_00270000.pickle ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p -p assets/reinfo/{nat,waveRNN}
  gdown --id 1-6fw29ePjeA_HcCw4pRuDOLAyPecRjLa -O assets/reinfo/nat/duration_ckpt_latest.pickle
  gdown --id 1co9Qe-exUfwiov5OVYjxrkUp1DUINZ5i -O assets/reinfo/nat/acoustic_ckpt_latest.pickle
  gdown --id 1-7ruVgwZC7mcdJZaR-qzV8CyLHhhuN7a -O assets/reinfo/waveRNN/checkpoint_00270000.pickle
fi
echo "Generate audio clip"
text=`cat assets/truyen_kieu.txt`
python3 -m vietTTS.synthesizer --text "$text" --output clip.wav --use-nat --lexicon-file assets/reinfo/lexicon.txt --use-nat --silence-duration 0.2