
from ._tools import *

requires_gym = skip_on_condition("not mdp.config.has_gym","This test requires OpenAi's Gym Library")

@requires_gym
def test_gym_nodes():
    gym_node = mdp.nodes.GymNode('MountainCar-v0', render=False, numx_rng=mdp.numx_rand.RandomState(seed=13))
    a = mdp.numx.array([[0], [0], [1]]).astype('float')
    out = gym_node(a)
    assert_array_equal(out[:,gym_node.observation_dim*2:gym_node.observation_dim*2+gym_node.action_dim], a)
    assert (out.shape[1] == gym_node.observation_dim*2+gym_node.action_dim+1+1)
    assert(gym_node.numx_rng == gym_node.env.np_random)

@requires_gym
def test_gym_continuous_explorer_node():

    node = mdp.nodes.GymContinuousExplorerNode('MountainCarContinuous-v0', epsilon=0., decay=1.0)
    a = node.get_random_actions(1000)
    out = node.execute(a)
    assert_array_equal(out[:,node.observation_dim*2:node.observation_dim*2+node.action_dim], a)

    node = mdp.nodes.GymContinuousExplorerNode('MountainCarContinuous-v0', epsilon=1., decay=0.0)
    out = node.execute(a)
    assert_array_equal(out[:,node.observation_dim*2:node.observation_dim*2+node.action_dim], a)
