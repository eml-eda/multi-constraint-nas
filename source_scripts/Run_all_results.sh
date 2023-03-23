#!/usr/bin/env bash

# _elements="mem_constraint+lat_obj"
# _loss="max"

# source Run_paper_results_only_size.sh PIT amd 0 &
# source Run_paper_results_only_size.sh Supernet amd 0 &

# source Run_paper_results_only_size.sh PIT icl 1 &
# source Run_paper_results_only_size.sh Supernet icl 1 &

# source Run_paper_results_only_size.sh PIT vww 2 &
# source Run_paper_results_only_size.sh Supernet vww 2 &

# source Run_paper_results_only_size.sh PIT kws 3 &
# source Run_paper_results_only_size.sh Supernet kws 2 &


# source Run_paper_results_size_latency.sh PIT icl 1 &
# source Run_paper_results_size_latency.sh Supernet icl 3 &

# source Run_paper_results_size_latency.sh PIT vww 2 &
# source Run_paper_results_size_latency.sh Supernet vww 3 &

# source Run_paper_results_size_latency.sh PIT kws 1 &


source Run_paper_results_size_latency_GAP8.sh PIT icl 1 &
source Run_paper_results_size_latency_GAP8.sh Supernet icl 3 &

source Run_paper_results_size_latency_GAP8.sh PIT vww 2 &
source Run_paper_results_size_latency_GAP8.sh Supernet vww 3 &

source Run_paper_results_size_latency_GAP8.sh PIT kws 1 &


# source Run_ablation_lambda_increasing.sh PIT icl 0 &
# source Run_ablation_obj_constr.sh PIT icl 1 &
# source Run_ablation_icv.sh Supernet icl 2 &
# source Run_ablation_abs.sh PIT kws 3 &