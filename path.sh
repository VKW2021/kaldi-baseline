export KALDI_ROOT=/apdcephfs/share_1157259/users/janinezhao/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
export PATH=$PATH:$KALDI_ROOT/tools/extras/kaldi_lm
export PATH=$PATH:$KALDI_ROOT/tools/F4DE/bin/

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
