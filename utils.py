import itertools

def vectorize(observation):
    if len(observation) == 3:
        ghost_positions = list(itertools.chain.from_iterable(observation[2]))
        return list(observation[0].flatten()) + list(observation[1]) + ghost_positions
    else:
        ghost_positions = list(itertools.chain.from_iterable(observation[1]))
        return list(observation[0]) + ghost_positions
