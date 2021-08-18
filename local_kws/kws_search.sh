#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.


help_message="$(basename $0): do keyword indexing and search.  data-dir is assumed to have
                 kws/ subdirectory that specifies the terms to search for.  Output is in
                 decode-dir/kws/
             Usage:
                 $(basename $0) <lang-dir> <data-dir> <decode-dir>"

# Begin configuration section.
#acwt=0.0909091
min_lmwt=7
max_lmwt=17
duptime=0.6
cmd=run.pl
model=
skip_optimization=false # true can speed it up if #keywords is small.
max_states=150000
indices_dir=
kwsout_dir=
stage=0
word_ins_penalty=0
extraid=
silence_word=  # specify this if you did to in kws_setup.sh, it's more accurate.
ntrue_scale=1.0
nbest=900
max_silence_frames=50
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -u
set -e
set -o pipefail


if [[ "$#" -ne "3" ]] ; then
    echo -e "$0: FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi

silence_opt=

langdir=$1
datadir=$2
decodedir=$3

if [ -z $extraid ] ; then
  kwsdatadir=$datadir/kws
else
  kwsdatadir=$datadir/${extraid}_kws
fi

if [ -z $kwsout_dir ] ; then
  if [ -z $extraid ] ; then
    kwsoutdir=$decodedir/kws
  else
    kwsoutdir=$decodedir/${extraid}_kws
  fi
else
  kwsoutdir=$kwsout_dir
fi
mkdir -p $kwsoutdir

if [ -z $indices_dir ]; then
  indices_dir=$kwsoutdir
fi

for d in "$datadir" "$kwsdatadir" "$langdir" "$decodedir"; do
  if [ ! -d "$d" ]; then
    echo "$0: FATAL: expected directory $d to exist"
    exit 1;
  fi
done
if [[ ! -f "$kwsdatadir/ecf.xml"  ]] ; then
    echo "$0: FATAL: the $kwsdatadir does not contain the ecf.xml file"
    exit 1;
fi

echo $kwsdatadir
duration=`head -1 $kwsdatadir/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    perl -e 'while($m=<>) {$m=~s/.*\"([0-9.]+)\".*/\1/; print $m/2;}'`

#duration=`head -1 $kwsdatadir/ecf.xml |\
#    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
#    grep -o -E "[0-9]*[\.]*[0-9]*" |\
#    perl -e 'while(<>) {print $_/2;}'`

echo "Duration: $duration"

if [ ! -z "$model" ]; then
    model_flags="--model $model"
else
    model_flags=
fi

frame_subsampling_factor=1
if [ -f $decodedir/../frame_subsampling_factor ] ; then
  frame_subsampling_factor=$(cat $decodedir/../frame_subsampling_factor)
  echo "Frame subsampling factor autodetected: $frame_subsampling_factor"
fi

if [ $stage -le 0 ] ; then
  if [ ! -f $indices_dir/.done.index ] ; then
    [ ! -d $indices_dir ] && mkdir  $indices_dir
    for lmwt in `seq $min_lmwt $max_lmwt` ; do
        indices=${indices_dir}_$lmwt
        mkdir -p $indices

        acwt=`perl -e "print (1.0/$lmwt);"`
        [ ! -z $silence_word ] && silence_opt="--silence-word $silence_word"
        steps/make_index.sh $silence_opt --cmd "$cmd" --acwt $acwt $model_flags\
          --skip-optimization $skip_optimization --max-states $max_states \
          --word-ins-penalty $word_ins_penalty --max-silence-frames $max_silence_frames\
          --frame-subsampling-factor ${frame_subsampling_factor} \
          $kwsdatadir $langdir $decodedir $indices  || exit 1
    done
    touch $indices_dir/.done.index
  else
    echo "Assuming indexing has been aready done. If you really need to re-run "
    echo "the indexing again, delete the file $indices_dir/.done.index"
  fi
fi


if [ $stage -le 1 ]; then
  for lmwt in `seq $min_lmwt $max_lmwt` ; do
      kwsoutput=${kwsoutdir}_$lmwt
      indices=${indices_dir}_$lmwt
      mkdir -p $kwsoutdir
      local_kws/search_index.sh --cmd "$cmd" --indices-dir $indices \
        --strict false --frame-subsampling-factor ${frame_subsampling_factor} \
        $kwsdatadir $kwsoutput  || exit 1

      nj=`cat $indices/num_jobs`
      #this will truncate the file
      rm -f $kwsoutput/results; touch $kwsoutput/results

      for job in `seq 1 $nj`; do
        {
          if [ -f $kwsoutput/result.${job}.gz ]; then
            gunzip -c $kwsoutput/result.${job}.gz
          else
            cat $kwsoutput/result.${job}
          fi
        } | cat - $kwsoutput/results | \
          local_kws/filter_kws_results.pl --nbest $nbest | \
          sort -u > $kwsoutput/results.${job}
        mv $kwsoutput/results.${job} $kwsoutput/results
      done

  done
fi

exit 0

