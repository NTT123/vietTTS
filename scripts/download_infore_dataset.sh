data_root=/tmp/infore # modify this variable

mkdir -p $data_root/{raw,processed}
pushd .
cd $data_root
gdown --id 1QsRZ2Pgorvn99teqhdSvezzamYIo0YQE -O infore.zip
unzip -P BroughtToYouByInfoRe infore.zip -d "$data_root/raw"
popd
cp assets/infore/infore_silence_spaces.txt $data_root/raw/pp/transcript.txt
