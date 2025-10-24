#!/bin/bash

architecture=infinity # infinity, instella_iar
case "$architecture" in
    "infinity")
        conda activate inf_water
        
        # INFINITY ARGUMENTS
        infinity2b=1
        if [ $infinity2b -eq 1 ]; then
            vae_path='/path/infinity/infinity_vae_d32reg.pth'
            model_path='/path/infinity/infinity_2b_reg.pth'
            checkpoint_type='torch'
            vae_type=32
            model='infinity_2b'
            apply_spatial_patchify=0
            batch_size=2
        else
            model='infinity_8b'
            checkpoint_type='torch_shard'
            model_path='/path/infinity/infinity_8b_weights'  # 8.4GB
            vae_type=14
            vae_path='/path/infinity/infinity_vae_d56_f8_14_patchify.pth'
            apply_spatial_patchify=1
            batch_size=1
        fi
        cfg=3
        tau=1
        rope2d_normalized_by_hw=2
        add_lvl_embeding_only_first_block=1
        rope2d_each_sa_layer=1
        text_channels=2048
        use_scale_schedule_embedding=0
        use_bit_label=1

        ;;
    "instella_iar")
        conda activate instella_new
        config='./Instella-T2I/configs/ar_config.py'
        ckpt_path='./Instella-T2I/checkpoints'
        ;;
    "big_r")
        conda activate BiGR
        image_size=512 # 256
        if [ $image_size -eq 256 ]; then
            image_size=256
            model='BiGR-L-d24' # Model name
            seq_len=256 # 256
            ckpt_bae='a'
            ckpt='a'
        else
            seq_len=1024 # 256
            ckpt_bae='/path/BiGR/pretrained_models/binaryae_ema_720000.th' # Path to the BAE checkpoint
            model='BiGR-L-512' # Model name
            ckpt='/path/BiGR/pretrained_models/bigr_L_d32.pt' # Path to the model checkpoint

        fi
        num_sample_iter=10 # Unmasking steps
        sampling_steps=10 # t
        gumbel_temp=0.01 # Temperature for Gumbel sampling
        cfg=2.5 # Configuration for the model
        ;;
    *)
        echo "Error: Unsupported architecture: $architecture"
        echo "Supported architectures: infinity, instella_iar"
        exit 1
        ;;
esac
echo USER: $USER
which conda
which python



# COMMON ARGUMENTS
batch_size=10
watermark_scales=2 #0: No watermark, 1: Only last scale, 2: Apply on all scales, 3: up to the 9th scale, 4: from the 10th scale up, 5: [3,4,5]
watermark_delta=$1 #Bias model towards green set
watermark_context_width=2 # When changed, make sure to check if it actually still works properly. This hasnt been touched in a while
watermark_gen_image=1 #0: Only detect watermark given save_file path, 1: generate image based on prompt and detect watermark
watermark_count_bit_loss_after_reencoding=1 # reencodes the generated image and counts the bits overlap. Can also count the token overlap, but the flag is hardcoded. Adds inference speed because the generated image has to be reencoded, so only use if needed
watermark_count_bit_flip=1
watermark_method='2-bit_pattern' # 2-bit_pattern or N-bit_pattern
out_dir="/path/${architecture}/${model}/delta_${watermark_delta}"
set="01,10"
dataset_path='/path/datasets/mscoco2014val' 
num_sampels=1000




# run the code
case "$architecture" in
    "infinity")
        echo "Running Infinity architecture"
        
        PYTHONPATH=. python3 ./comprehensive_infer.py \
        --cfg ${cfg} \
        --tau ${tau} \
        --pn ${pn} \
        --model_path ${model_path} \
        --vae_type ${vae_type} \
        --vae_path ${vae_path} \
        --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
        --use_bit_label ${use_bit_label} \
        --model_type ${model_type} \
        --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
        --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
        --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
        --tau ${tau} \
        --checkpoint_type ${checkpoint_type} \
        --text_channels ${text_channels} \
        --apply_spatial_patchify ${apply_spatial_patchify} \
        --seed 0 \
        --max_samples ${num_samples} \
        --watermark_scales ${watermark_scales} \
        --watermark_delta ${watermark_delta} \
        --watermark_context_width ${watermark_context_width} \
        --watermark_gen_image ${watermark_gen_image} \
        --watermark_count_bit_loss_after_reencoding ${watermark_count_bit_loss_after_reencoding} \
        --watermark_method ${watermark_method} \
        --watermark_count_bit_flip ${watermark_count_bit_flip} \
        --set "${set}" \
        --out_dir "${out_dir}" \
        --dataset_path "${dataset_path}" \
        --batch_size ${batch_size} \
        --architecture ${architecture} \
        --enable_model_cache 0 # if finetuned, enable this 
        ;;
    "instella_iar")
        echo "Running instella IAR architecture"
        PYTHONPATH=. python3 ./comprehensive_infer.py \
        --seed 0 \
        --max_samples ${num_samples} \
        --watermark_scales ${watermark_scales} \
        --watermark_delta ${watermark_delta} \
        --watermark_context_width ${watermark_context_width} \
        --watermark_gen_image ${watermark_gen_image} \
        --watermark_count_bit_loss_after_reencoding ${watermark_count_bit_loss_after_reencoding} \
        --watermark_method ${watermark_method} \
        --watermark_count_bit_flip ${watermark_count_bit_flip} \
        --set "${set}" \
        --out_dir "${out_dir}" \
        --dataset_path "${dataset_path}" \
        --batch_size ${batch_size} \
        --architecture ${architecture} \
        --config ${config} \
        --ckpt_path ${ckpt_path} \
        ;;
    "big_r")
        echo "Running BiGR architecture"
        PYTHONPATH=. python3 ./comprehensive_infer.py \
        --seed 0 \
        --max_samples ${num_samples} \
        --watermark_scales ${watermark_scales} \
        --watermark_delta ${watermark_delta} \
        --watermark_context_width ${watermark_context_width} \
        --watermark_gen_image ${watermark_gen_image} \
        --watermark_count_bit_loss_after_reencoding ${watermark_count_bit_loss_after_reencoding} \
        --watermark_method ${watermark_method} \
        --watermark_count_bit_flip ${watermark_count_bit_flip} \
        --set "${set}" \
        --out_dir "${out_dir}" \
        --dataset_path "imagenet" \
        --batch_size ${batch_size} \
        --architecture ${architecture} \
        --image_size ${image_size} \
        --seq_len ${seq_len} \
        --ckpt_bae ${ckpt_bae} \
        --model ${model} \
        --ckpt ${ckpt} \
        --num_sample_iter ${num_sample_iter} \
        --gumbel_temp ${gumbel_temp} \
        --cfg ${cfg} \
        --infer_steps ${sampling_steps} \
        ;;
    *)
        echo "Error: Unsupported architecture: $architecture"
        echo "Supported architectures: infinity, instella_iar"
        exit 1
        ;;
esac


