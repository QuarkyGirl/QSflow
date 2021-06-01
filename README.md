# QSflow

QSflow is a Python library for computing FRG flows of the [quasi-stationary effective action](https://arxiv.org/abs/2104.10687) for false vacuum decay problems.

## Installation

QSflow requires [Python 3](https://www.python.org/downloads/) and has been tested with 3.8.2 and 3.9.5.

Use the [pip](https://pip.pypa.io/en/stable/) package manager to install the python dependencies

```bash
pip3 install -r requirements.txt
```

The examples require [LaTeX](https://www.latex-project.org/get/) by default;
this can be disabled by commenting out the line `rc('text', usetex=True)` in `plot_h.py`.

## Usage

For an arbitrary scalar potential, use the classes `Scalar` (zero-temperature) or `Scalar_T` (finite temperature).

```python
from scalar_h import Scalar
φ,k = ... [field and k values over which to evaluate] ...
V = lambda x: ... [your potential] ...
scalar = Scalar(V)
sol = scalar.flow(φ,k)
```

For a specific potential (e.g. φ^4 potentials) it is preferable to use the one of the pre-written subclasses, e.g.

```python
from scalar_h import Phi3
m2,α,λ = ... [your couplings] ...
scalar = Phi3(m2,α,λ)
sol = scalar.flow(φ,k)
```

## Examples

To run the example files:

```bash
python3 example_N.py
```

Current examples:
 - `example_1.py`: QSEA and perturbative effective potential for a φ^3 theory at
 zero temperature
 - `example_2.py`: Effective potentials in the QSEA, perturbation theory, and
 unmodified FRG for a φ^3 theory. Somewhat more computationally intensive due to
 stiffness of the unmodified FRG

## License
[ACSL](https://anticapitalist.software/)
