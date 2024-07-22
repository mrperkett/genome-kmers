# Introduction

This Python package implements objects and functions that allow for fast and memory-efficient [*k*-mer](https://en.wikipedia.org/wiki/K-mer) calculations on the genome.  See the [genome-kmers homepage](https://genome-kmers.readthedocs.io/en/latest/index.html) for a detailed overview, examples, dev setup, and API reference.

This package can be installed via PyPI

```shell
pip install genome-kmers
```

or via cloning the repo and installing (e.g. using [poetry](https://python-poetry.org/))

```bash
git clone git@github.com:mrperkett/genome-kmers.git
cd genome-kmers/

poetry install

# run tests to verify everything is working properly
poetry run python -m pytest
```