# Improving Hyperparameter Optimization By Planning Ahead
We provide here the source code for our paper: [Dataset2Vec: Learning Dataset Meta-Features](We provide here the source code for our paper: [Improving Hyperparameter Optimization By Planning Ahead](https://arxiv.org/abs/2110.08028).
).

## Usage
To meta-train the joint surrogate model, run the run-meta.py file.
```
python run-meta.py 
```

Use the weights with the best validation performance to initialize the surrogate for hyperparameter optimization.

```
python test-pets.py
```

## Citing LookAhead-MPC
-----------

To cite LookAhead-MPC please reference our arXiv paper:


```
@article{jomaa2021improving,
  title={Improving Hyperparameter Optimization by Planning Ahead},
  author={Jomaa, Hadi S and Falkner, Jonas and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2110.08028},
  year={2021}
}
```
