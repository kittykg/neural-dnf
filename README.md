# Neural DNF-based Models

A collection of neural DNF-based models based on semi-symbolic layers first
introduced in the pix2rule [1].

## Models

The following models are implemented in `neural_ndnf/neural_dnf.py`:

**Neural DNF** (`NeuralDNF`): The vanilla neural DNF model from pix2rule [1],
consisting of a conjunctive semi-symbolic layer and a disjunctive neural layer,
both with tanh activation.

**Neural DNF-EO** (`NeuralDNFEO`): An extension of the neural DNF model from
[2], with a frozen conjunctive layer as a constraint layer after the disjunctive
layer.

**Neural DNF-MT** (`NeuralDNFMutexTanh`): Our new model introduced in the paper
[3], consisting of a conjunctive semi-symbolic layer with tanh activation and a
disjunctive neural layer with the mutex-tanh activation (implemented as a
function `mutex_tanh(...)` in `neural_ndnf/common.py`). The semi-symbolic layer
with mutex-tanh activation is implemented as class `SemiSymbolicMutexTanh` in
`neural_ndnf/semi_symbolic.py` .

## Post-training Processing

The post-training processing that extracts logical interpretation from a trained
neural DNF-based model is implemented in `neural_ndnf/post_training.py`.

The novel disentanglement method described in our paper 'Disentangling Neural
Disjunctive Normal Form Models' [4] is implemented in the function
`split_entangled_conjunction(...)` in `neural_ndnf/post_training.py`.

## How to Use

### Install

Install this package in editable mode:

```bash
pip install -e . --config-settings editable_mode=strict
```

The config setting flag is used for resolving Pylance's issue with importing.

### Unit testing

To run the unit test, run the following command in the root directory of the
project:

```bash
python -m unittest discover -p "*_test.py"
```

## References

[1] Cingillioglu, N., & Russo, A. (2021). pix2rule: End-to-end Neuro-symbolic
Rule Learning. In A. S. D. Garcez & E. Jiménez-Ruiz (Eds.), Proceedings of the
15th International Workshop on Neural-Symbolic Learning and Reasoning as part of
the 1st International Joint Conference on Learning & Reasoning (IJCLR 2021),
Virtual conference, October 25-27, 2021 (pp. 15–56). Retrieved from
https://ceur-ws.org/Vol-2986/paper3.pdf

[2] Baugh, K. G., Cingillioglu, N., & Russo, A. (2023). Neuro-symbolic Rule
Learning in Real-world Classification Tasks. In A. Martin, H.-G. Fill, A.
Gerber, K. Hinkelmann, D. Lenat, R. Stolle, & F. van Harmelen (Eds.),
Proceedings of the AAAI 2023 Spring Symposium on Challenges Requiring the
Combination of Machine Learning and Knowledge Engineering (AAAI-MAKE 2023),
Hyatt Regency, San Francisco Airport, California, USA, March 27-29, 2023.
Retrieved from https://ceur-ws.org/Vol-3433/paper12.pdf

[3] Kexin Gu Baugh, Luke Dickens, and Alessandra Russo. 2025. Neural DNF-MT: A
Neuro-symbolic Approach for Learning Interpretable and Editable Policies. In
Proc. of the 24th International Conference on Autonomous Agents and Multiagent
Systems (AAMAS 2025), Detroit, Michigan, USA, May 19 – 23, 2025, IFAAMAS.
https://dl.acm.org/doi/10.5555/3709347.3743538

[4] Kexin Gu Baugh, Vincent Perreault, Matthew Baugh, Luke Dickens, Katsumi
Inoue, and Alessandra Russo. 2025. Disentangling Neural Disjunctive Normal Form
Models. Coming in NeSy 2025. Arxiv pre-print: https://arxiv.org/abs/2507.10546
