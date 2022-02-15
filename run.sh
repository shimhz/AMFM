#1
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_2_2_fixed_mixup \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_2_2 -model_name Model \
-comet_disable True

#2
CUDA_VISIBLE_DEVICES=1 python -W ignore main.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_1_fixed_mixup \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_1 -model_name Model \
-comet_disable True

#1
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_1_fixed_mixup_specaug \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_1 -model_name Model \
-comet_disable True

#============================================================================
#2
CUDA_VISIBLE_DEVICES=1 python -W ignore main_rnw.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_1_specaug_fixed_mixup5 \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_1 -model_name Model \
-comet_disable True

#1
CUDA_VISIBLE_DEVICES=0 python -W ignore main_rnw.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_1_specaug_fixed_mixup_chk \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_1 -model_name Model \
-comet_disable True

#3
CUDA_VISIBLE_DEVICES=2 python -W ignore main_rnw.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_1_specaug_fixed_mixup_modified \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_1_modified -model_name Model \
-comet_disable True

#4
CUDA_VISIBLE_DEVICES=3 python -W ignore main_rnw.py \
-name amfm2_cirriculum_warmup_cosine_mel256_specmix4_1_specaug_fixed_mixup_modified1 \
-warm True \
-lr_decay cosine \
-ratio 5 \
-model_scp amfm_cirriculum2_extend_group4_1_modified1 -model_name Model \
-comet_disable True