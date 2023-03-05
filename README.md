# PinSAGE
pinsage for wine recommendation
This is the PinSAGE package applied to the wine recommendation system prepared by the 11th Tobigs Conference "투믈리에".
It was implemented based on the DGL library and modified to fit the project in the PinSAGE example.

PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf <br>
DGL: https://docs.dgl.ai/# <br>
DGL PinSAGE example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage <br>

## Requirements

- - dgl
- - dask
- - pandas
- - torch
- - torchtext
- - sklearn

## Dataset

### Vivino
11,900,000 Wines & 42,000,000 Users
User feature: userID, user_follower_count, user_rating_count
Item feature: wine_id, body, acidity, alcohol, rating_average, grapes_id


We have a request to share data, so we provide it for you to use in part.
* 100,000 review data
* User Metadata
* Wine Metadata

As much as it's not the entire data, when you learn it yourself, performance may not come out as much as you want.
**process_wine.py** is code that preprocesses collected data for DGL If you use the data provided, please refer to it.


## Training model

### Nearest-neighbor recommendation

This model recommends wine as Knearst Neighbors for all users.
This method finds the center of the embedding vector of the wine consumed by a specific user and recommends the K wines closest to the center vector.

```
python model.py -d data.pkl -s model -k 500 --eval-epochs 100 --save-epochs 100 --num-epochs 500 --device 0 --hidden-dims 128 --batch-size 64 --batches-per-epoch 512
```

- d: Data Files
- s: The name of the model to
- k: top K count
- eval epochs: performance output epoch interval (0 = output X)
- save epochs: storage epoch interval (0 = storage X)
- - num epochs: epoch 횟수
- hidden dims: embedding dimension
- batch size: batch size
- - batches per epoch: iteration 횟수

In addition, there are parameters applied by PinSAGE, so please refer to the model.py code.

## Inference
The code at the bottom is a code that explains how to infer, and I will explain the train function part of model.py by excerpt.
The performance evaluation method for this project differs from the traditional DGL PinSAGE recommendation of only one item.

### Embeddings
Since the model is aimed at learning the embedding of nodes, it is necessary to obtain embeddings of all items and then perform similarity measurements or clustering separately through vector-to-vector operations.

```
model.py line 159

h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], data_dict['testset'], item_ntype, neighbor_sampler, args.batch_size, device)
```
Obtain all embeddings by receiving node information from the DGL graph object. shape becomes (number of users, embedding size).

```
model.py line 182~
h_center = torch.mean(h_nodes, axis=0) # central embedding
dist = h_center @ h_item.t() # center embedding * all embeddings -> matrix product
topk = dist.topk(args.k)[1].cpu().extract k in numpy() # dist size order
```
We average the node embeddings of a particular user for inference to obtain a central embedding vector, and obtain Distance with all embeddings and matrix operations.
We extract as many as K embeddings in the order of small distances and present them as final recommendations.

Evaluate with Recall and Hitrate whether the selected items belong to the verification data.

## Performance

Model | Hitrate | Recall
------------ | ------------- | -------------
SVD | 0.854 | 0.476
PinSAGE | 0.942 | 0.693
