# Neural DNF-based Models

Neural DNF-based models, using semi-symbolic layers based on pix2rule's vanilla
neural DNF model [1].

We also include the extended model neural DNF-EO (NDNF-EO) from [2].

Our new model: neural DNF with mutex-tanh activation (NDNF-MT) model is
implemented as class `NeuralDNFMutexTanh` in `neural_ndnf/neural_dnf.py` .

The mutex-tanh activation is implemented as a function `mutex_tanh(...)` in
`neural_ndnf/common.py` .

The semi-symbolic layer with mutex-tanh activation is implemented as class
`SemiSymbolicMutexTanh` in `neural_ndnf/semi_symbolic.py` .

## How to Use

Install this package in editable mode:

```bash
pip install -e . --config-settings editable_mode=strict
```

The config setting flag is used for resolving Pylance's issue with importing.

## Unit testing

To run the unit test, run the following command in the root directory of the
project:

```bash
python -m unittest discover -p "*_test.py"
```

## TODOs

* README.md: Add structure section

* Add testing for `SemiSymbolicMutexTanh`

* Add testing for `NDNF-MT`

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
