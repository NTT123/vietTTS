data_dir="./train_data" # modify this

mkdir -p "$data_dir"

if [ ! -f $data_dir/LJSpeech-1.1.tar.bz2 ]; then
  pushd `pwd`
  cd $data_dir
  wget -q --show-progress "https://drive.google.com/u/0/uc?id=1s8KF_zOBFRpMRPDmSGsHQ67uSwtEXYqP&export=download" -O ljs_textgrid.zip
  wget -q --show-progress https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  tar xjf LJSpeech-1.1.tar.bz2
  cd LJSpeech-1.1
  wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -O lexicon.txt
  cd wavs
  unzip ../../ljs_textgrid.zip
  for file in wav-*.TextGrid; do mv "$file" "${file/wav-/}"; done
  popd
fi