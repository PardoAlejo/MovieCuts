#!/bin/bash
# DIR=/ibex/ai/home/pardogl/LTC-e2e
# cd $DIR

# # LRs=(0.01 0.03 0.05 0.07 0.09 0.1 0.3)
# LRs=(0.09 0.1 0.3)
# focal_ons=(0 1)
# # logit_neg_scales=(2.0)
# # logit_init_biass=(0.05)
# map_alphas=(0.1)
# map_betas=(10.0)
# map_gammas=(0.2 0.3)
# reweight_func='rebalance'
# weight_norms=('by_instance' 'by_batch')
# echo Running $reweight_func
# for LR in ${LRs[@]}; do
#   for focal_on in ${focal_ons[@]}; do
#     for weight_norm in ${weight_norms[@]}; do
#       for map_gamma in ${map_gammas[@]}; do
#         echo ${LR} ${focal_on} ${weight_norm} ${map_gamma} ${reweight_func}
#         export LR focal_on weight_norm map_gamma reweight_func
#         sbatch scripts/run_db_loss_rebalance.sh
#       done
#     done
#   done
# done

LRs=(0.03)
focal_ons=(0)
CB_betas=(0.3 0.5 0.7)  
CB_mode='average_w' #options ['by_class', 'average_n', 'average_w', 'min_n']
logit_neg_scales=(1.0)
logit_init_biass=(0.1)
reweight_func='CB'
weight_norm='by_batch'
echo Running $reweight_func
for LR in ${LRs[@]}; do
  for focal_on in ${focal_ons[@]}; do
    for CB_beta in ${CB_betas[@]}; do
      for logit_neg_scale in ${logit_neg_scales[@]}; do
        for logit_init_bias in ${logit_init_biass[@]}; do
            echo ${LR} ${focal_on} ${CB_beta} ${CB_mode} ${weight_norm} ${reweight_func} ${logit_init_bias} ${logit_neg_scale}
            export LR focal_on CB_beta CB_mode weight_norm reweight_func logit_init_bias logit_neg_scale
            sbatch scripts/run_db_loss_CB.sh
        done
      done
    done
  done
done