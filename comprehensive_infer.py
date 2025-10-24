# Geneate images from prompts, and detect watermark from generated images

import os
import json
import logging
logging.basicConfig(encoding="utf-8", level=logging.WARNING)
logger = logging.getLogger(__name__)
import argparse
from os import path as osp
import numpy as np
import torch
import traceback
from tqdm import tqdm
import detect_watermark
from detect_watermark import WatermarkInference, get_detector
from helper import set_seeds,save_single_image, count_matching_n_bit_sequences, save_images, get_stripped_delta, count_match_after_reencoding
from datasets import get_gen_dataset
torch._dynamo.config.cache_size_limit = 64
from pydantic.utils import deep_update
import random
from architecture_wrapper import ArchitectureWrapper, get_architecture, VAEWrapper, get_vae, get_architecture_arguments

def process_short_text(short_text):
    if '--' in short_text:
        processed_text = short_text.split('--')[0]
        if processed_text:
            short_text = processed_text
    return short_text

lines4infer = []

if __name__ == "__main__":
    
    parser = get_architecture_arguments()  # Pass architecture to get specific args
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="/path")
    parser.add_argument("--save_gen_bits", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()
    set_seeds(args.seed)
    out_dir = args.out_dir 
    stripped_delta = get_stripped_delta(args.watermark_delta)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"save to {out_dir}")

    dataset_path = args.dataset_path
    metadataset, infer_type = get_gen_dataset(dataset_path, num_samples=args.max_samples)
    image_duplicates = []
    cnt = 0
    skipped_images = 0

    for annotation in metadataset:
        if cnt== args.max_samples:
            break
        if annotation["image_id"] in image_duplicates:
            continue
        image_duplicates.append(annotation["image_id"])

        prompt = annotation["caption"]
        if not prompt and prompt != 0:
            continue
        image_path = osp.join(out_dir, f"{annotation['image_id']}.png")
        if args.watermark_gen_image:
            if os.path.exists(image_path):
                skipped_images +=1
                cnt+=1
                continue
        else:
            if not os.path.exists(image_path):
                skipped_images +=1
                continue
        
        lines4infer.append(
            {
                "image_id": annotation["image_id"],
                "prompt_id": annotation["prompt_id"],
                "prompt": prompt,
                "h_div_w": 1.0,
                "infer_type": infer_type,
                'image_path': image_path
            }
        )
        cnt+=1

    print(f"Skipped {skipped_images} images as these already existed")
    
    print(f"Totally {len(lines4infer)} items for infer")

    vae_wrapper : VAEWrapper = get_vae(args)

    model_wrapper : ArchitectureWrapper = get_architecture(args, vae_wrapper)  


    jsonl_list = []
    infer_full_data = {}
    cnt = 0
    watermark_inference = WatermarkInference(args, vae_wrapper)
    if hasattr(watermark_inference, "scales"):
        scales = watermark_inference.scales
    else:
        scales = None
    if watermark_inference.message == None:
        message = None
    else:
        message = watermark_inference.message[0,...]
    watermark_detector = get_detector(args, message)
    if args.watermark_gen_image:
        jsonl_file = f"{out_dir}/metrics.json"
    else:
        jsonl_file = f"{out_dir}/metrics_detect.json"

    batch_size = args.batch_size

    batches = [lines4infer[i:i+batch_size] for i in range(0, len(lines4infer),batch_size)]
    inference_time = []
    detection_time = []
    green_fraction = []
    for i, batch in enumerate(tqdm(batches, miniters=50)):
        try:
            logging.info(batch)
            prompts = []
            for entry in batch:
                prompt = entry["prompt"]
                #prompt = process_short_text(prompt)
                prompts.append(prompt)

            
            if args.watermark_gen_image:
                with torch.no_grad():
                    ret, gen_bit_indices, image = model_wrapper.gen_img(
                        prompts,
                        vae_wrapper.vae,
                        watermark_inference
                    )
                
                #inference_time.append(ret['time'])
                save_single_image(image, [entry["image_path"] for entry in batch])
            for i in range(len(batch)):
                metadata = batch[i]
                if args.save_gen_bits:
                    if args.architecture == "infinity":
                        gen_bits = []
                        #Gen bit indices is list of tensor with [bs, 1, h, w, d]
                        for j in range(len(gen_bit_indices)):
                            gen_bits.append(gen_bit_indices[j][i,...])
                    else:
                        gen_bits = gen_bit_indices[i]
                    torch.save(gen_bits, metadata['image_path'].replace('.png', '_gen_bits.pt'))

                detect_results = detect_watermark.detect(args, metadata['image_path'], watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=scales, detect_on_each_scale = False)
                #detection_time.append(detect_results["time"])
                metadata["z_score"] = round(detect_results["z_score"],5)
                metadata["green_fraction"] = round(detect_results["green_fraction"], 3)
                metadata["stat_data"] = {} 

                if args.watermarn_gen_image:
                    ret_count = vae_wrapper.calc_bit_overlap(gen_bit_indices, metadata['image_path'], i, scales)
                    metadata["stat_data"] = deep_update(metadata["stat_data"], ret[i]["stat_data"])
                    metadata["stat_data"] = deep_update(metadata["stat_data"], ret_count)
                metadata["stat_data"] = deep_update(metadata["stat_data"], detect_results["stat_data"])
                jsonl_list = json.dumps(metadata, default=str) + "\n"
                with open(jsonl_file, 'a') as f:
                    f.writelines(jsonl_list)        
        except Exception as e:
            logger.warning(f"{e}", traceback.print_exc())
            logger.warning(f"Error at batch {i}: {batch}")
    tmp_inf = inference_time
    tmp_det = detection_time
    if len(tmp_inf)>1000:
        inference_time = tmp_inf[:1000]
        detection_time = tmp_det[:1000]
    else:
        inference_time = tmp_inf
        detection_time = tmp_det
    
    print("Timing without warmup")
    print(f"Time/Image: {np.mean(inference_time)} ({np.std(inference_time)}); max time: {max(inference_time)}, min time: {min(inference_time)}")
    print(f"Detection/Image: {np.mean(detection_time)} ({np.std(detection_time)}), max time: {max(detection_time)}, min time: {min(detection_time)}")

    print("Timing with warmup")
    if len(tmp_inf)>50:
        inference_time = tmp_inf[50:]
        detection_time = tmp_det[50:]
        print(f"Time/Image: {np.mean(inference_time)} ({np.std(inference_time)}); max time: {max(inference_time)}, min time: {min(inference_time)}")
        print(f"Detection/Image: {np.mean(detection_time)} ({np.std(detection_time)}), max time: {max(detection_time)}, min time: {min(detection_time)}")