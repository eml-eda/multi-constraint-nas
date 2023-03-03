#!/usr/bin/env bash

# _elements="mem_constraint+lat_obj"
# _loss="max"

source Run_paper_results_only_size.sh PIT amd 0 &
source Run_paper_results_only_size.sh Supernet amd 0 &

source Run_paper_results_only_size.sh PIT icl 1 &
source Run_paper_results_only_size.sh Supernet icl 1 &

source Run_paper_results_only_size.sh PIT vww 2 &
source Run_paper_results_only_size.sh Supernet vww 2 &

source Run_paper_results_only_size.sh PIT kws 3 &
source Run_paper_results_only_size.sh Supernet kws 3 &
