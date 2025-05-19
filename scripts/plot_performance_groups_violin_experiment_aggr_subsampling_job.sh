#!/usr/bin/env bash

~/src/dmriseg/scripts/plot_performance_groups_violin_experiment_aggr_subsampling.sh \
  dwi \
  -c \
  dwisub20 \
  dwisub30 \
  dwisub60

~/src/dmriseg/scripts/plot_performance_groups_violin_experiment_aggr_subsampling.sh \
  dwi1k \
  -c \
  dwi1ksub20 \
  dwi1ksub30 \
  dwi1ksub60
