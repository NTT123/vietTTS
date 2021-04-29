data_root=/tmp/reinfo # modify this variable

mkdir -p $data_root/{raw,processed}
pushd .
cd $data_root
gdown --id 1QsRZ2Pgorvn99teqhdSvezzamYIo0YQE
unzip -P BroughtToYouByInfoRe reinfo.zip -d "$data_root/raw"
popd
cp assets/reinfo/reinfo_silence_spaces.txt $data_root/raw/pp/transcript.txt
