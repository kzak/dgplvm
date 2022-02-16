# Discriminative GPLVM

An implementation of Discriminative GPLVM
based on [Utram et al, 2007](https://dl.acm.org/doi/abs/10.1145/1273496.1273613?casa_token=ZoR2sCkaWwkAAAAA:nRw0IHevdVWLqbD7JufdVAObtzM0lE0cuLi_vHH-RgtyHqvBkIiaAUsOQjFNlJ4SL18Xb96yRYTkjFQ).


## Requirement
- Python 3.9
- Other packages are listed in [requirements.txt](https://github.com/kzak/dgplvm/requirements.txt) .


## Installation

```sh
git clone https://github.com/kzak/dgplvm.git
cd dgplvm

python -m venv venv
source ./venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```


## Usage
Please see "notebooks/dgplvm_example.ipynb".

```sh
cd /path/to/dgplvm
source ./venv/bin/activate

jupyter lab
# Open "notebooks/dgplvm_example.ipynb" in jupyter lab
```


## Note
Because of my naive implementation, 
- There may be lots of bugs.
- Optimization of latent space and kernel parameters is still very unstable.

If you find any bugs, please let me know.
