# Reproducing watermark in the sand attack

## Acknowledgement
The code for reproducing watermarking in the sand attack is built on [the Github implementation](https://github.com/hlzhang109/impossibility-watermark/tree/main) of the paper "Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models". Thanks for their great work!

## Install dependencies
Please refer to their instructions for installing the dependencies.

## Launch the attack
1. Change `[WATERMARKED_IMAGE_DIR]` in `./cv_attack.py` to your directory of watermarked images.

2. Run the following command to reproduce the watermark in the sand attack with slurm:
```bash
sbatch ./run_attack.sh
```

3. Detect the watermark with the watermarking detection script.

