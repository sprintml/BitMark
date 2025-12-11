
# BitMark: Watermarking Bitwise Autoregressive Image Generative Models

This is the official PyTorch code for the paper: 

[**BitMark: Watermarking Bitwise Autoregressive Image Generative Models**](https://neurips.cc/virtual/2025/poster/117685)

*NeurIPS 2025*

**TL; DR**: We introduce a bitwise, radioactive watermark for bitwise image generative models and show that it remains robust against various attacks while preserving high image quality and generation speed.

## Abstract

State-of-the-art text-to-image models like Infinity generate photorealistic images at an unprecedented speed. These models operate in a bitwise autoregressive manner over a discrete set of tokens that is practically infinite in size. However, their impressive generative power comes with a growing risk: as their outputs increasingly populate the Internet, they are likely to be scraped and reused as training data—potentially by the very same models. This phenomenon has been shown to lead to model collapse, where repeated training on generated content, especially from the models’ own previous versions, causes a gradual degradation in performance. A promising mitigation strategy is watermarking, which embeds human-imperceptible yet detectable signals into generated images—enabling the identification of generated content. In this work, we introduce BitMark, a robust bitwise watermarking framework. Our method embeds a watermark directly at the bit level of the token stream during the image generation process. Our bitwise watermark subtly influences the bits to preserve visual fidelity and generation speed while remaining robust against a spectrum of removal techniques. Furthermore, it exhibits high radioactivity, i.e., when watermarked generated images are used to train another image generative model, this second model’s outputs will also carry the watermark. The radioactive traces remain detectable even when only fine-tuning diffusion or image autoregressive models on images watermarked with our BitMark. Overall, our approach provides a principled step toward preventing model collapse in image generative models by enabling reliable detection of generated outputs.

## Preparation

Please first download the pretrained weights for the respective models by following the READMEs in the corresponding folder. 

### Code Structure

BitMark is defined in extended_watermark_processor.py. 

The robustness evaluation is in tools/robustness_test.py and to run it, it requires a path of clean images and a path of watermarked images and then computes the TPR@1%FPR for all images in the watermarked path against all attacks.   

Our novel BitFlipper attack is applied by running flipper.py


## Infinity

Infinity generates in a per scale fashion, where each scale can be understood as corresponding to a different resolution.


### uv setup

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies in the root folder of this project:

```bash
uv sync
```

### Running the model

If you want to run the Jupyter notebook starter (recommended), specify `bitmark` as environment in `./scripts/infer_infinity.ipynb`.

To download the required model weights, run:

```bash
uv run download_models.py --output-dir './weights'
```

Otherwise, the file tools/comprehensive_infer.py offers a way to run the Infinity model and watermark images using our BitMark. The file can be executed by calling: 

```
uv run tools/comprehensive_infer.py --model_path "./weights/Infinity/infinity_2b_reg.pth" --vae_type 32 --vae_path "./weights/Infinity/infinity_2b_reg.pth" --add_lvl_embeding_only_first_block 1 --model_type "infinity_2b" --seed 0 --watermark_scales 2 --watermark_delta 2 --watermark_context_width 2 --out_dir "./" --jsonl_filepath = "captions.json"
```

where --jsonl_filepath leads to a json, which contains the prompts used for image generation. The delta and sequence length (watermark_context_width) can be adapted if wished.  


## Cite our work

If you find our work useful, please cite our paper:
```bibtex
@inproceedings{
  kerner2025bitmark,
  title={BitMark: Watermarking Bitwise Autoregressive Image Generative Models},
  author={Louis Kerner and Michel Meintz and Bihe Zhao and Franziska Boenisch and Adam Dziedzic},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=VSir0FzFnP}
}
```


## License

The code is licensed under an MIT license. It relies on code and models from other repositories. See the next Acknowledgements section for the licenses of those dependencies.

## Acknowledgements

This project builds on the following repositories:
- [Infinity](https://github.com/FoundationVision/Infinity)
- [A Watermark for Large Language Models](https://github.com/jwkirchenbauer/lm-watermarking)
- [Instella IAR](https://github.com/AMD-AGI/Instella-T2I)
- [BiGR](https://github.com/haoosz/BiGR)
- [Watermarks in the Sand](https://github.com/hlzhang109/impossibility-watermark)

We thank these great works!
