# Beltrami Flow & Neural Diffusion on Graphs (BLEND) for Graph Classification
This repository is an implementation / slight adaptation of the [Beltrami Flow and Neural Diffusion on Graphs (BLEND) model proposed by Chamberlain et al. (2021)](https://arxiv.org/pdf/2110.09443.pdf). The [source code](https://github.com/twitter-research/graph-neural-pde) mainly focuses on node classification, while this repository uses BLEND for graph classification on the ShapeNet dataset.

```
@article
{chamberlain2021blend,
  title={Beltrami Flow and Neural Diffusion on Graphs},
  author={Chamberlain, Benjamin Paul and Rowbottom, James and Eynard, Davide and Di Giovanni, Francesco and Dong Xiaowen and Bronstein, Michael M},
  journal={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021, Virtual Event},
  year={2021}
}
```

## Experiments
Due to the large quantity of the ShapeNet dataset compared to the original datasets for node classification, I only got to run the model on a couple categories of ShapeNet so far.

For example, to test the model on the "Cap" and "Rocket" categories of ShapeNet, run the following command:

```
python gen_pos_encodings.py --shapenet_data Cap,Rocket
```
