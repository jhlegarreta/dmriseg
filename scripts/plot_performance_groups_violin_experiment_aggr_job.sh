#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

measure_names=("dice" "hd95" "msd" "cm_dist" "vs" "lab_detect_rate")

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv
in_performance_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures/performance_plots_aggr/

for _measure_name in "${measure_names[@]}"; do

  echo "Measure name:" ${_measure_name}

  python /home/jhlegarreta/src/dmriseg/scripts/plot_performance_groups_violin.py \
    --in_performance_dirnames \
    ${in_performance_dirname}/dmri_hcp_t1/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_b0/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_sphm_b1000-2000-3000/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_sphm_b1000/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_sphm_b2000/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_sphm_b3000/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_fa/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_md/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_rd/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_evals_e1/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_evals_e2/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_evals_e3/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_ak/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_mk/aggregate_performance \
    ${in_performance_dirname}/dmri_hcp_rk/aggregate_performance \
    --out_dirname ${out_dirname} \
    --measure_name ${_measure_name} \
    --in_labels_fname ${lut_fname} \
    --aggregate_only

done
