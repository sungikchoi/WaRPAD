# Code of Training-free Detection of AI-generated images via Cropping Robustness.

## Requirements

### Environment setting

We refer to our library in requirements.txt. 

### Downloading Dataset

We use three benchmarks for the main experiment. Due to the size, we have provided a download link for the dataset. 

GenImage: http://github.com/GenImage-Dataset/GenImage

Synthbuster: https://ieeexplore.ieee.org/document/10334046 , https://loki.disi.unitn.it/RAISE/download.html

Deepfake-LSUN-Bedroom: http://github.com/jonasricker/diffusion-model-deepfake-detection

After downloading the code, change the address on the utils.py 

## Testing the code

For the replication of our WaRPAD in GenImage, 

Run `python final_imgn.py` 

We also provide the test code for AEROBLADE and RIGID, which were used in the experiment.





