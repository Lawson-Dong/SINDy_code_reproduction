# SINDy Code Reproduction

This repository contains the complete code implementation for my undergraduate thesis:  
**"Discovering Physics from Data: A Review of SINDy and Symbolic Learning Methods"**.

- **Thesis Abstract**: [A place for hyperlink of the paper]

- **Main Experiments**: Replication of SINDy, DSINDy, SINDy-PI, and Ensemble SINDy on Lorenz system and constrained pendulum dynamics.

- **How to Run**: Simply open and execute the Jupyter notebooks in order.  
  All notebooks are self-contained and include step-by-step demonstrations.

## Repository Structure

| Notebook File | Description |
| :--- | :--- |
| `SINDy_fitting_Lorenz_System.ipynb` | Apply standard SINDy to identify dynamics of the Lorenz 63 system. |
| `SINDy_and_DSINDy.ipynb` | Compare robustness of standard SINDy vs. DSINDy under different noise levels (1%–25%). |
| `SINDy_and_SINDy_PI.ipynb` | Compare performance on explicit (polar pendulum) vs. implicit (Cartesian constrained pendulum) systems. |
| `ESINDy.ipynb` | Implement Ensemble SINDy with bagging and analyze sensitivity to the sparsity threshold λ. |

## Requirements

The code is written in Python 3 and relies on standard scientific computing libraries.  
To install the required packages, run:

```bash
pip install numpy scipy matplotlib pysindy
