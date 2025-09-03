import random
import numpy as np
from typing import List, Tuple
from transition import (
    Observation,
    Expression,
    Generate_New_Expression,
    Likelihood
)
from mcmc import generate_expression_from_prior

def smc(o: List[Observation], M: int, J: int, n: int = 1) -> Tuple[List[List[Expression]], List[List[float]]]:
    """
    Args:
        o: observations, [O_0, O_1, ..., O_J]
        M: number of particles
        J: number of iterations
        n: number of rejuvenations
    Returns:
        particles: particle at every time step
        weights: weight at every time step
    """

    particles = [[] for _ in range(J + 1)]
    weights = [[] for _ in range(J + 1)]
    
    # S0
    for l in range(M):
        while True:
            E_0_l = generate_expression_from_prior()
            if Likelihood(o[0], E_0_l) > 0:
                particles[0].append(E_0_l)
                weights[0].append(1.0)
                break
    # S1-S3
    for j in range(1, J + 1):
        O_j = o[j]
        O_jm1 = o[j - 1]
        
        # S1 Reweight
        new_weights = []
        for l in range(M):
            E_prev = particles[j-1][l]
            likelihood_j = Likelihood(O_j, E_prev)
            likelihood_jm1 = Likelihood(O_jm1, E_prev)
            
            w_j_ell = likelihood_j / likelihood_jm1 if likelihood_jm1 > 0 else 0
            new_weights.append(w_j_ell)
        
        # Normalization
        weight_sum = sum(new_weights)
        if weight_sum > 0:
            new_weights = [w / weight_sum for w in new_weights]
        else:
            new_weights = [1.0 / M] * M
        
        # S2 Resample
        if j < J:
            resampled_indices = systematic_resample(new_weights, M)
            resampled_particles = [particles[j-1][idx] for idx in resampled_indices]
            particles[j] = resampled_particles
            weights[j] = [1.0] * M
        else:
            # final: save instead of resample
            particles[j] = particles[j-1].copy()
            weights[j] = new_weights
        
        # S3 Rejuvenate
        if n > 0:
            for l in range(M):
                current_expr = particles[j][l]
                # n times mcmc
                for _ in range(n):
                    current_expr = Generate_New_Expression(O_j, current_expr)
                particles[j][l] = current_expr
    
    return particles, weights

def systematic_resample(weights: List[float], M: int) -> List[int]:
    """
    Args:
        weights
        M: number of particles
    Returns:
        indices of resampled list
    """
    indices = []
    cumulative_weights = np.cumsum(weights)
    step = 1.0 / M
    u = random.uniform(0, step)
    
    i = 0
    for m in range(M):
        while u > cumulative_weights[i] and i < M - 1:
            i += 1
        indices.append(i)
        u += step
    
    return indices