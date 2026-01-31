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
To reproduce the entire code, please use``` python TensoGraph\Tensograph_oneil.py ```to run oneil dataset.
Please use``` python TensoGraph\Tensograph_cloud.py ```to run cloud dataset.
Please pay attention to modifying and saving the paths of the parameters and the dataset in this section.

For the visualized code of the ONEIL dataset: (Note to change the paths for different scores, for example, in the ONEIL dataset, the relevant sections such as #oneil loewe or #oneil bliss or #oneil hsa or #oneil zip should be commented out) 

For the PCC and RMSE values of the oneil dataset in all cell lines, please use TensoGraph\all_PCC.py and TensoGraph\all_RMSE.py respectively. 

For the PCC and RMSE values of the oneil dataset in each cell line, please use TensoGraph\every_PCC.py and TensoGraph\every_RMSE.py respectively.




|Argument|Default|Description|
|---|---|----|
| learning_rate|  1e-4|  Initial learning rate. |
| epochs|  If the data for Oneil is 500, and if it's the data for ALMANAC and Cloud, it would be 200.|  The number of training epochs. |
| embedding_dim|  320|  The number of dimension for drug embeddings. |
| dropout|  If the Oneil dataset does not have a dropout layer, and if the ALMANAC and Cloud datasets have a dropout rate of 0.2|  Dropout rate (1 - keep probability) |
| weight_decay|  0|  Weight for L2 loss on embedding matrix. |
| val_test_size|  0.1|  the rate of validation and test samples. |

