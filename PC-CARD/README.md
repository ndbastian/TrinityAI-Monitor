
## Generate the COCO-OOC dataset:
Run `generate_coco_ooc_*.py (e.g., generate_coco_ooc_vehicle_in_indoor.py`) to generate OOC images and annotation files for each OOC categories in the COCO-OOC dataset.

## Dependencies:
Tested with Python 3.7.6 and the anaconda environment file used for the project is included as `environment.yaml`.

## Train/Test GCRN Model: 

1. Run `GMNN/train_coco_ooc.py` to train models.
2. Run `GMNN/test_coco_ooc.py` to test GCRN.

## Data: 

Required paths
data_root   : /workspace/datasets/coco_ooc/graphs_normalized_part
graph models: /workspace/datasets/coco_ooc/models/extra_gmm_save
OOC images  : /workspace/datasets/coco_ooc/OOC_images
OOC outputs : /workspace/datasets/coco_ooc/OOC_images_outputs
 
## Sample python command:

python test_coco_ooc.py --data_root /workspace/datasets/coco_ooc/graphs_normalized_part --model_dir /workspace/datasets/coco_ooc/models/extra_gmm_save --img_ip_dir /workspace/datasets/coco_ooc/OOC_images --img_op_dir /workspace/datasets/coco_ooc/OOC_images_outputs

## Notebook

Demo.ipynb: Chnage the paths in cell [3] and run cell [4]