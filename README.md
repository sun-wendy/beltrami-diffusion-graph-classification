# BLEND for Graph Classification
This repository is an implementation / slight modification of the [Beltrami Flow and Neural Diffusion on Graphs (BLEND) model proposed by Chamberlain et al. (2021)](https://arxiv.org/pdf/2110.09443.pdf). The [official source code](https://github.com/twitter-research/graph-neural-pde) mainly focuses on node classification, while this repository uses BLEND for graph classification on the ShapeNet dataset. Most of the code is directly copied from the official code.


## Requirements
The code has been tested running under Python 3.7.13.


## Main Modifications from Official BLEND
- `models/data.py` includes the ShapeNet dataset
- `gen_pos_encodings.py` generates positional encodings for ShapeNet and saves them in data/pos_encodings
- `models/GNN.py` changes from node classification to graph classification task


## Experiments
Due to the large quantity of the ShapeNet dataset compared to the original datasets for node classification, I only got to run the model on a couple categories of ShapeNet so far.

For example, to test the model on the "Cap" and "Rocket" categories of ShapeNet, run the following command:

```
python gen_pos_encodings.py --shapenet_data Cap,Rocket
```


## Performance
- Test accuracy of **~73.23%** on the single category of "Cap"


## Further Work
- Test the model on the entire ShapeNet dataset, rather than a few categories


## Reference
Chamberlain, B.P., Rowbottom, J., Eynard, D., Giovanni, F.D., Dong, X., & Bronstein, M.M. (2021). Beltrami Flow and Neural Diffusion on Graphs. *Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021*. https://doi.org/10.48550/arXiv.2110.09443
```
@article
{chamberlain2021blend,
  title={Beltrami Flow and Neural Diffusion on Graphs},
  author={Chamberlain, Benjamin Paul and Rowbottom, James and Eynard, Davide and Di Giovanni, Francesco and Dong Xiaowen and Bronstein, Michael M},
  journal={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021, Virtual Event},
  year={2021}
}
```


## License
The official repository is under [Apache License 2.0](https://github.com/twitter-research/graph-neural-pde/blob/main/LICENSE).
