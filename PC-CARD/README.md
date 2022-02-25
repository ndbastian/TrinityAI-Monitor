

# Predictive Coding based Context-awaRe novelty Detection (PC-CARD)

### One of the components of TrinityAI is PC-CARD: Predictive Coding based Context-aware Novelty Detection. PC-CARD uses the Layer 2 of the TrinityAI monitoring framework currently implemented via Graph Neural Networks to identify when the input has no novel concept but the concepts appears in a new composition with out-of-context concept(s). We use object detection task to demonstrate this capability here. In this task, out-of-context concepts correspond to instances of a known object class but occurring in a new context. For example, a typically outdoor concept such as a zebra occurring indoors in the reception of an office.

![alt text](https://github.com/SRI-CSL/Trinity-AI/blob/main/assets/ooc-zebra.jpg?raw=true)

## Train Object Detector: 

1. Run `frcnn/train_bettercoco_distributed.py` to train models on COCO object detection dataset.


## Train & test Graph Models: 

1. To generate image-wise bounding box features run  `extract_graph_features_COCO.py` for COCO or `extract_graph_features_OOCD.py` for the OOOC dataset.   This saves pickle files for each input images to a folder. 
	* To compute stats i.e. mean and variance for feature normalization, run `compute_stats.py`.   
3.  To build graphs from the features run `build_graphs.py`  with directory as CLI input. 
4. Run `gcn_train.py` to train and `gcn_test.py` files to train or test graph models. Find the saved models in the log directory.


## Mahalanobis Distance(MD):

1. Run `gcn_mahalanobis.py` to extract mid-level features and save class-wise features  from trained models.
2. Run `mahalanobis/compute_robust_cov.py` to generate class wise covariance matrices  and min/max distances for each class.
3. Run `mahalanobis/compute_robust_cov.py` to get MD  scores for new samples.
4. To generate ROC curves run  `notebooks/mahalanobis.ipynb ` file with  correct filepaths.

## Scripts:
Scripts to run all these tests are also found under `scripts` folder.
For example you can do:

`python scripts/train.sh` to train any model.



