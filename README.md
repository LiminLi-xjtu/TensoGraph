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
To reproduce the entire code, please use: python codes\drug_tucker+gtn.py
Please pay attention to modifying and saving the paths of the parameters and the dataset in this section.


To reproduce the relevant ablation experiments, please use:
1. python codes\drug_tucke_gtn-A.py    

This code is used to remove the molecular structural features of the drug.

2. python codes\drug_tucke_gtn-A-tucker.py

This code is used to remove the molecular structure features of the drug and obtain the features obtained through the Tucker decomposition part.

3. python codes\drug_tucke_gtn-A-gtn.py

This code is used to remove the molecular structure features of the drug and the features of the GTN part, thereby obtaining the final characteristics.


4. python codes\drug_tucker_gtn-B.py
This code is used to remove the features obtained from the Tucker decomposition and the GTN part.This code is used to remove the features obtained from the Tucker decomposition and the GTN part.

对于可视化的代码：（注意更改不同得分上的路径，比如ONEIL数据集将#oneil loewe或者# oneil bliss或者# oneil hsa或者# oneil zip的相关部分进行注释）

对于oneil数据集在所有细胞系中的PCC和RMSE请分别用codes\all_PCC_hezitu.py和codes\all_RMSE_hezitu.py

对于oneil数据集在每个细胞系中的PCC和RMSE请分别用codes\every_PCC_zhexiantu.py和codes\every_RMSE_zhexiantu.py






|Argument|Default|Description|
|---|---|----|
| learning_rate|  1e-4|  Initial learning rate. |
| epochs|  If the data for Oneil is 500, and if it's the data for ALMANAC and Cloud, it would be 200.|  The number of training epochs. |
| embedding_dim|  320|  The number of dimension for drug embeddings. |
| dropout|  If the Oneil dataset does not have a dropout layer, and if the ALMANAC and Cloud datasets have a dropout rate of 0.2|  Dropout rate (1 - keep probability) |
| weight_decay|  0|  Weight for L2 loss on embedding matrix. |
| val_test_size|  0.1|  the rate of validation and test samples. |

