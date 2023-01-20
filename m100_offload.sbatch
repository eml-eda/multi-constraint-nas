#!/bin/bash

sbatch  --account IscrC_AdHeNAS --job-name icl-training-$1-$2-$3-$4-$5-$6-$7 \
	--partition m100_usr_prod \
	--time 24:00:00 --nodes 1 --cpus-per-task 8 --gres gpu:1 --mem 60GB \
	--mail-user alessio.burrello@polito.it --mail-type ALL \
	--output $HOME/Alessio/multi-objective-nas/log/%x.log \
	--error $HOME/Alessio/multi-objective-nas/err_log/%x.log \
	$HOME/Alessio/multi-objective-nas/run_flexnas.sh icl $1 $2 $3 $4 $5 $6 $7
