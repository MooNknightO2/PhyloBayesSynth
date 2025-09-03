from typing import List, Optional
import random

from transition import (
    Observation,
    Expression,
    Generate_New_Expression,
    Likelihood
)

def generate_expression_from_prior() -> Expression:
    raise NotImplementedError("not implemented") # to be realized

def bayesian_synthesis_mcmc(o: Observation, n: int) -> List[Expression]:
    """
    MCMC for Bayesian synthesis
    Args:
        observation: O
        n_iterations: n ≥ 1
    Returns:
        list: E₁, ..., Eₙ
    """
    while True:
        E_0 = generate_expression_from_prior()
        if Likelihood(o, E_0) > 0:
            break

    samples = []
    current_expr = E_0
    
    for i in range(n):
        new_expr = Generate_New_Expression(o, current_expr)
        samples.append(new_expr)
        current_expr = new_expr
    return samples