import random
from typing import Any, Tuple

# to be realized
class Nonterminal:
    pass

class Expression:
    pass

class Observation:
    pass

def Uniform(distribution: list) -> Any:
    return random.choice(distribution)

def Sever(expression: Expression, node: Any) -> Tuple[Nonterminal, Expression]:
    raise NotImplementedError("not implemented")

def Expand(nonterminal: Nonterminal) -> Expression:
    raise NotImplementedError("not implemented")

def Likelihood(expression: Expression, observation: Observation) -> float:
    raise NotImplementedError("not implemented")

def getNodes(expression: Expression):
    raise NotImplementedError("not implemented")


# transition operator
def Generate_New_Expression(O: Observation, E: Expression) -> Expression:
    nodes_A_E = getNodes(E) 
    a = Uniform(nodes_A_E)
    
    N_i, E_sev = Sever(E, a)
    E_sub = Expand(N_i)
    E_prime = E_sev.fill_hole(E_sub) # fill_hole method to be realized

    L = Likelihood(E, O)
    L_prime = Likelihood(E_prime, O)

    nodes_A_E_prime = getNodes(E_prime)
    accept_prob = min(1, (len(nodes_A_E) / len(nodes_A_E_prime)) * (L_prime / L))

    r = random.uniform(0, 1)
    if r < accept_prob:
        return E_prime
    else:
        return E