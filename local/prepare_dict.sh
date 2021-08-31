#!/usr/bin/env bash
set -exo
. ./path.sh

dir=data/local/dict

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dict-dir>";
  exit 1;
fi

dir=$1

python3 local/mk_nonsilence_phones.py $dir/lexicon.txt $dir/nonsilence_phones.txt

echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

cat $dir/silence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;


echo "local/prepare_dict.sh succeeded"
exit 0;
