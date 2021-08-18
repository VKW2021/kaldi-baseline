#!/usr/bin/env bash

trn_set=
dev_set=
tst_set=

lexicon=data/vkw/lexicon.txt
nj=20
stage=1
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  mkdir -p data/local/dict 
  cp $lexicon data/local/dict || exit 1;
  local/prepare_all.sh ${trn_set} ${dev_set} ${tst_set} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh --nj $nj --stage $gmm_stage
fi

exit 0;
