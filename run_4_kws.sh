#!/usr/bin/env bash

stage=0
dataset=dev_lgv
src_dir=exp/chain/tdnn_cnn_tdnnf_1a_ft_15hs_combined_ft_15hs_combined_sp
score_files_dir=data/vkw/score # which has ecf.xml, rttm
kwlist_files_dir=data/vkw/keyword # which has kwlist.xml

min_lmwt=8
max_lmwt=12
cmd=run.pl
max_states=150000
wip=0.5 #Word insertion penalty
iter=final


decode_dir=$src_dir/decode_$dataset
score_files_dir=$score_files_dir/$dataset

lang_dir=data/lang
model=`dirname $decode_dir`/${iter}.mdl

# End configuration section.                                                    
echo "$0 $@"  # Print the command line for logging


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh
set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will

if [ $stage -le 1 ]; then
    echo KWS prepare
    mkdir -p data/$dataset/kws
    for f in rttm utter_map; do
        cp $score_files_dir/$f data/$dataset/kws || exit 1
    done
    #cp $kwlist_files_dir/kwlist.xml data/$dataset/kws || exit 1
    ### we use our modified kwlist.xml hear
    
    cp data/$dataset/kwlist.xml data/$dataset/kws || exit 1
    cp $score_files_dir/ecf data/$dataset/kws/ecf.xml || exit 1
    ./local_kws/kws_data_prep.sh data/lang data/$dataset data/$dataset/kws || exit 1
    #exit 0
fi

if [ $stage -le 2 ]; then
    echo KWS search
    if [ ! -f $decode_dir/.done.kws ] ; then
      local_kws/kws_search.sh --cmd "$cmd" \
        --max-states ${max_states} --min-lmwt ${min_lmwt} \
         --max-lmwt ${max_lmwt} --indices-dir $decode_dir/kws_indices \
         --model $model \
        $lang_dir data/$dataset $decode_dir || exit 1
      touch $decode_dir/.done.kws
    fi
 
fi
exit 0
