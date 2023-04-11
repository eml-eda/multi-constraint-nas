#!/usr/bin/env bash

# _elements="mem_constraint+lat_obj"
# _loss="max"

source Run_size_latency.sh $1 True 0 mem_constraint+lat_obj max MaxvsAbs &
source Run_size_latency.sh $1 True 1 mem_constraint+lat_obj abs MaxvsAbs &

source Run_size_latency.sh $1 True 2 mem_constraint+lat_obj max Gumbel &
source Run_size_latency.sh $1 False 3 mem_constraint+lat_obj max Gumbel &

source Run_size_latency.sh $1 True 0 mem_constraint max MemConstrvsObj &
source Run_size_latency.sh $1 True 1 mem_obj max MemConstrvsObj &

source Run_size_latency.sh $1 True 2 mem_constraint+lat_obj max LatConstrvsObj &
source Run_size_latency.sh $1 True 3 mem_constraint+lat_constraint max LatConstrvsObj &

source Run_lambda_increasing.sh $1 True 2 mem_constraint max LambdaIncreasing increasing &
source Run_lambda_increasing.sh $1 True 3 mem_constraint max LambdaIncreasing const &