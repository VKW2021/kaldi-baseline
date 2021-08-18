#!/bin/bash

#for y in "lgv" "liv" "stv"; do
for y in "lgv"; do
   kws_results_dir=exp/chain/tdnn_cnn_tdnnf_1a_ft_15hs_combined_ft_15hs_combined_sp/decode_dev_${y}/kws_8
   bash scripts/bin/results_to_score.sh data/vkw/score/dev_${y}/ecf \
       data/vkw/label/lab_${y}/dev_5h/segments \
       data/vkw/score/dev_${y}/utter_map \
       $kws_results_dir/results \
       data/vkw/keyword/kwlist.xml \
       data/vkw/score/dev_${y}/rttm || { echo "Error computing TWV"; exit -1; }
   bash scripts/bin/F1.sh $kws_results_dir/kws_outputs/f4de_scores_unnormalized/alignment.csv > $kws_results_dir/kws_outputs/f4de_scores_unnormalized/F1.txt
done
