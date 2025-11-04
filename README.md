# TensoGraph
This is the Python code for the paper "TensoGraph: A Tensor–Transformer Framework for Global–Local Drug Synergy Prediction on Heterogeneous Graphs".

## Requirements
Python 3.8 or higher  

pandas 1.3.5

numpy 1.21.2

tensorflow 2.4.1    


## Datasets
1. O'Neil dataset
2. ALMANAC dataset
3. CLOUD datset


## Training
To reproduce the entire code, please use: python TensoGraph\drug_tucker+gtn.py
Please pay attention to modifying and saving the paths of the parameters and the dataset in this section.


To reproduce the relevant ablation experiments, please use:
1. python TensoGraph\drug_tucke_gtn-A.py    

This code is used to remove the molecular structural features of the drug.

2. python TensoGraph\drug_tucke_gtn-A-tucker.py

This code is used to remove the molecular structure features of the drug and obtain the features obtained through the Tucker decomposition part.

3. python TensoGraph\drug_tucke_gtn-A-gtn.py

This code is used to remove the molecular structure features of the drug and the features of the GTN part, thereby obtaining the final characteristics.


4. python TensoGraph\drug_tucker_gtn-B.py
This code is used to remove the features obtained from the Tucker decomposition and the GTN part.This code is used to remove the features obtained from the Tucker decomposition and the GTN part.


For the visualized code: (Note to change the paths for different scores, for example, in the ONEIL dataset, the relevant sections such as #oneil loewe or #oneil bliss or #oneil hsa or #oneil zip should be commented out) 
For the PCC and RMSE values of the oneil dataset in all cell lines, please use TensoGraph\all_PCC_hezitu.py and TensoGraph\all_RMSE_hezitu.py respectively. 
For the PCC and RMSE values of the oneil dataset in each cell line, please use TensoGraph\every_PCC_zhexiantu.py and TensoGraph\every_RMSE_zhexiantu.py respectively.




|Argument|Default|Description|
|---|---|----|
| learning_rate|  1e-4|  Initial learning rate. |
| epochs|  If the data for Oneil is 500, and if it's the data for ALMANAC and Cloud, it would be 200.|  The number of training epochs. |
| embedding_dim|  320|  The number of dimension for drug embeddings. |
| dropout|  If the Oneil dataset does not have a dropout layer, and if the ALMANAC and Cloud datasets have a dropout rate of 0.2|  Dropout rate (1 - keep probability) |
| weight_decay|  0|  Weight for L2 loss on embedding matrix. |
| val_test_size|  0.1|  the rate of validation and test samples. |

