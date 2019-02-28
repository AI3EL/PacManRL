import itertools
import numpy as np


# To comply with keras format : flatten and transform in np.array
def to_array(observations):
    # high_dim
    lst = []
    for observation in observations:
        lst.append(vectorize(observation))
    return np.array(lst)


def vectorize(observation):
    if len(observation) == 3:
        ghost_positions = list(itertools.chain.from_iterable(observation[2]))
        return list(observation[0].flatten()) + list(observation[1]) + ghost_positions
    else:
        ghost_positions = list(itertools.chain.from_iterable(observation[1]))
        return list(observation[0]) + ghost_positions
