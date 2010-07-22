import mdp

SEED = None

def _compare_with_seed(seed):
    global SEED
    if SEED is None:
        SEED = seed
    return SEED == seed

def test_seed():
    seed = mdp.numx_rand.get_state()[1][0]
    assert _compare_with_seed(seed)
    mdp.numx_rand.seed(seed+1)

def test_seed_clone():
    # we need two identical functions to check that the seed
    # is reset at every call
    seed = mdp.numx_rand.get_state()[1][0]
    assert _compare_with_seed(seed)
    mdp.numx_rand.seed(seed+1)

