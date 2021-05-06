if [ ! -f assets/infore/waveRNN/checkpoint_00270000.pickle ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p -p assets/infore/{nat,waveRNN}
  gdown --id 1-6fw29ePjeA_HcCw4pRuDOLAyPecRjLa -O assets/infore/nat/duration_ckpt_latest.pickle
  gdown --id 1co9Qe-exUfwiov5OVYjxrkUp1DUINZ5i -O assets/infore/nat/acoustic_ckpt_latest.pickle
  gdown --id 1-TU4kYgTeevPl3Nng4F0F6nXrPUwJX8n -O assets/infore/waveRNN/checkpoint_00300000.pickle
fi
echo "Generate audio clip"
text=`cat assets/truyen_kieu.txt`
python3 -m vietTTS.synthesizer --text "$text" --output clip.wav --use-nat --lexicon-file assets/infore/lexicon.txt --use-nat --silence-duration 0.2