# Rotation estimation method

Our watermarking can be detected with a high TPR at 1% FPR on rotated images after applying the following rotation estimation method to invert the rotated images.
The code for reproducing this method is built on [the Github implementation](https://github.com/pidahbus/deep-image-orientation-angle-detection/tree/main) of the paper "Deep Image Orientation Angle Detection". Thanks for their great work!

## Install dependencies
Please refer to their instructions for installing the dependencies. We note that the original dependencies might be vulnerable to version conflicts, so we also provide the `requirements.txt` used for our successful setup.

## Estimate the rotation angle and rotate back
We provide the jupyter notebook for estimating the rotation angle and rotating the image to the correct orientation here:

```./notebooks/inference.ipynb```

Please change `[ROTATED_IMAGE_DIR]` in the notebook to your directory of rotated images.


