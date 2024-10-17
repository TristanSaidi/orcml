# *ORC-ManL*: Recovering Manifold structure using Ollivier-Ricci Curvature

## Introduction
This repository contains the official implementation of ORC-ManL, an algorithm that uses discrete graph curvature to prune nearest-neighbor graphs. Our paper presenting this algorithm can be viewed here: https://arxiv.org/abs/2410.01149.

<div align="center">
  <img src="https://github.com/TristanSaidi/orcml/blob/main/demo.gif" alt="Demo GIF">
</div>

## Requirements
Please install libraries in the requirements.txt file using the following command:
```
pip install -r requirements.txt
```
Also be sure to adjust your PYTHONPATH,
```
export PYTHONPATH=/path/to/repository
```

## Evaluation
Scripts for evaluation tasks are provided in `official_experiments/`. To run the pruning evaluation for example, simply run the following command:
```
python official_experiments/pruning.py
```

#### Real Data Procurement
For some experiments, data was obtained from [this paper](https://zenodo.org/records/3357167). To recreate experiments, please extract the `MouseV1_MouseALM_HumanMTG/` directory into `./data/`.
 

## Example Usage
Here is an example to illustrate how one may use our implementation. We provide plotting code for visualizing graphs in `src/plotting.py`.

```
from src.orcml import *
from src.data import *
from src.plotting import *

data_dictionary = concentric_circles(n_points=4000, factor=0.385, noise=0.1)
data = data_dictionary['data']
labels = data_dictionary['cluster']

# algorithm and nearest neighbor parameters
params = {
    'mode': 'nbrs', # 'nbrs' or 'radius'
    'n_neighbors': 20, # number of neighbors
    'delta': 0.8, # confidence parameter: used to compute ORC thresh
    'lda': 0.01, # confidence parameter: used to compute metric distortion thresh
}

orcmanl = ORCManL(
    exp_params=params,
)
orcmanl.fit(data)
G_pruned = orcmanl.G_pruned

# visualize the graph
plot_graph_2D(X=data, graph=G_pruned, title='Pruned Graph')
```
