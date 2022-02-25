
## Generate the COCO-OOC dataset:
Run `generate_coco_ooc_*.py (e.g., generate_coco_ooc_vehicle_in_indoor.py`) to generate OOC images and annotation files for each OOC categories in the COCO-OOC dataset.

## Dependencies:
Tested with Python 3.7.6 and the anaconda environment file used for the project is included as `environment.yaml`.

## Train/Test GCRN Model: 

1. Run `GMNN/train_coco_ooc.py` to train models.
2. Run `GMNN/test_coco_ooc.py` to test GCRN.

## Data: 

1. data_root: /workspace/aroy/datasets/coco_ooc/
2. OOC images: /workspace/aroy/datasets/coco_ooc/OOC_images/
3. Val data: /workspace/aroy/datasets/coco_ooc/graphs_normalized_part
4. graph models: /workspace/aroy/datasets/coco_ooc/models/extra_gmm_save/
5. sample OOC outputs: /workspace/aroy/datasets/coco_ooc/OOC_images_outputs/
