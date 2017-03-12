import mdp
from difflib import SequenceMatcher
import gym
from gym import spaces
import time


class GymNode(mdp.OnlineNode):
    """GymNode is a thin OnlineNode wrapper over OpenAi's Gym library.
    For more information about the Gym library refer to the
    documentation provided at https://gym.openai.com/

    This node is a non-trainable node. The node's execute call takes
    an action array as an input and returns a flattened array of
    (current observation (observation), future observation, action, reward, done flag).

    The node's metaclass is selected to be an OnlineNode instead of a Node
    in order to enforce a shared numx_rng between the node and the gym's
    environment.

    The node also provides additional utility methods:
    'stop_rendering' - closes gym's rendering window if initialized.

    'get_environment_samples' - generates a required number of output
    samples for randomly selected actions, starting from the current env state.
    Note that the env state changes after the method is called.
    This is equivalent to an execute call without any input.

    'get_random_actions' - returns a random set of valid actions
    within the environment.

    'reset' - resets the environment.


    **Instance variables of interest**

      ``self.observation_dim / self.action_dim``
         Flattened observation  / action dimension

      ``self.observation_shape / self.action_shape``
         The original observation shape / action shape. Eg. (100,80,3) for an RGB image.

      ``self.observation_type / self.action_type``
         Discrete or continuous observation / action space.

      ``self.observation_lims / self.action_lims``
         Upper and lower bounds of observation / action space.

      ``self.n_observations / self.n_actions``
         Number of observations or actions for discrete types.

    """

    def __init__(self, env_name, render=False, render_interval=1, auto_reset=True, dtype=None, numx_rng=None):
        """
        env_name - Registered gym environment name. Eg. "MountainCar-v0"
        render - Enable or disable rendering. Disabled by default.
        render_interval - Number of execution steps delay between rendering
        auto_reset - Automatically resets the environment if gym env's done is True.
        """
        super(GymNode, self).__init__(input_dim=None, output_dim=None, dtype=dtype, numx_rng=None)

        self._env_registry = gym.envs.registry.env_specs
        if env_name in self._env_registry.keys():
            self._env_name = env_name
        else:
            similar_envs_str = '(' + ', '.join(self._get_similar_env_names(env_name)) + ')'
            raise mdp.NodeException(
                "Unregistered environment name. Are you looking for any of these?: \n%s" % similar_envs_str)
        self.env = gym.make(self.env_name)

        self.render = render
        self.auto_reset = auto_reset

        # set a shared numx_rng
        self.numx_rng = numx_rng

        # get observation dims and shape
        if isinstance(self.env.observation_space, spaces.discrete.Discrete):
            self.observation_type = 'discrete'
            self.observation_dim = 1
            self.observation_shape = (1,)
            self.n_observations = self.env.observation_space.n
            self.observation_lims = [[0], [self.n_observations - 1]]
        else:
            self.observation_type = 'continuous'
            self.observation_shape = self.env.observation_space.shape
            self.observation_dim = mdp.numx.product(self.observation_shape)
            self.n_observations = None
            self.observation_lims = [self.env.observation_space.low, self.env.observation_space.high]

        # get action dims
        if isinstance(self.env.action_space, spaces.discrete.Discrete):
            self.action_type = 'discrete'
            self.action_dim = 1
            self.action_shape = (1,)
            self.n_actions = self.env.action_space.n
            self.action_lims = [[0], [self.n_actions - 1]]
        else:
            self.action_type = 'continuous'
            self.action_shape = self.env.action_space.shape
            self.action_dim = mdp.numx.product(self.action_shape)
            self.n_actions = None
            self.action_lims = [self.env.action_space.low, self.env.action_space.high]

        # set input_dim
        self._input_dim = self.action_dim

        # set output dims
        self._output_dim = self.observation_dim * 2 + self.action_dim + 1 + 1

        # get observation
        self._phi = mdp.numx.reshape(self.env.reset(), [1, self.observation_dim])

        self.render_interval = render_interval
        self._interval = 1 if self.render_interval == -1 else self.render_interval
        self._flow_time = 0
        self._cnt = 0
        self._epicnt = 0

    # properties

    @property
    def env_name(self):
        """Returns the environment name"""
        return self._env_name  # read only

    def _get_similar_env_names(self, name):
        keys = self._env_registry.keys()
        ratios = [SequenceMatcher(None, name, key).ratio() for key in keys]
        return [x for (y, x) in sorted(zip(ratios, keys), reverse=True)][:5]

    def _set_numx_rng(self, rng):
        """Set a shared numx random number generator.
        """
        self.env.np_random = rng
        self._numx_rng = rng

    # Node capabilities

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    def _render_step(self):
        self._cnt += 1
        if self.render:
            _flow_dur = time.time() - self._flow_time
            if self._cnt % int(self._interval) == 0:
                t = time.time()
                self.env.render()
                _plot_dur = time.time() - t
                if self.render_interval == -1:
                    self._interval *= (100 * _plot_dur / _flow_dur + (self._cnt / self._interval - 1) *
                                       self._interval) / float(self._cnt)
                    self._interval = mdp.numx.clip(self._interval, 1, 50)
            self._flow_time = time.time()

    # environment steps
    def _steps(self, x):
        for a in x:
            if self.action_type == 'discrete':
                a = int(mdp.numx.asscalar(a))
            phi, r, done, info = self.env.step(a)
            self._render_step()
            if self.auto_reset and done:
                self.env.reset()
                self._epicnt += 1
            yield phi, a, r, done, info

    def _execute(self, x):
        x = mdp.numx.clip(x, self.action_lims[0], self.action_lims[1])
        phi_, a, r, done, info = zip(*self._steps(x))
        phi_ = mdp.numx.reshape(phi_, [len(phi_), self.observation_dim])
        phi = mdp.numx.vstack((self._phi, phi_[:-1]))
        self._phi = phi_[-1:]
        a = mdp.numx.reshape(a, [len(a), self.action_dim])
        r = mdp.numx.reshape(r, [len(r), 1])
        done = mdp.numx.reshape(done, [len(done), 1])
        y = mdp.numx.hstack((phi, phi_, a, r, done))
        return self._refcast(y)

    def _train(self, x):
        pass

    # public utility methods

    def stop_rendering(self):
        """stop gym's rendering if active"""
        self.env.render(close=True)

    def get_random_actions(self, n=1):
        """Returns the specified number of randomly (uniform) sampled actions."""
        return self._refcast(mdp.numx.reshape([self.env.action_space.sample() for _ in xrange(n)], (n, self.input_dim)))

    def get_environment_samples(self, n=1):
        """Returns observations corresponding to the specified number of randomly (uniform) sampled actions."""
        return self.execute(self.get_random_actions(n))

    def get_current_episode(self):
        """Returns the current epsiode."""
        return self._epicnt

    def get_current_iteration(self):
        """Returns the numbers of environment steps that have been executed so far"""
        return self._cnt

    def reset(self):
        """Resets the environment, agent's position, the current iteration count and the epsiode count."""
        self.env.reset()
        self._cnt = 0
        self._epicnt = 0
        self._interval = 1 if self.render_interval == -1 else self.render_interval
        self._flow_time = 0


class GymContinuousExplorerNode(GymNode):
    """
    Gym node that uses a continuous-action epsilon-greedy strategy to modulate between the given action and
    actions generated via a random walk.

    Only supports continuous action spaces
    """

    def __init__(self, env_name, epsilon=1.0, decay=1.0, action_std=None, action_momentum=0.,
                 render=False, render_interval=1, auto_reset=False, dtype=None, numx_rng=None):

        self.epsilon = epsilon
        self._decay = decay
        if self.decay < 1.:
            self._is_trainable = True
        else:
            self._is_trainable = False

        super(GymContinuousExplorerNode, self).__init__(env_name, render=render, render_interval=render_interval,
                                                        auto_reset=auto_reset, dtype=dtype, numx_rng=numx_rng)
        if self.action_type == 'discrete':
            raise mdp.NodeException("'GymContinuousExplorerNode supports only for 'continuous' actions, "
                                    "given 'discrete'.")

        self._cov = mdp.numx.identity(self.action_dim) if action_std is None else mdp.numx.diag(action_std)
        self._m = action_momentum
        self._a = mdp.numx.zeros(self.action_dim)
        self._init_epsilon = epsilon

    def is_trainable(self):
        return self._is_trainable

    @property
    def decay(self):
        """Return decay. Read only parameter"""
        return self._decay

    def _valid_action(self, a):
        if hasattr(self.env, 'valid_action'):
            return self.env.valid_action(a)
        else:
            return self.env.action_space.contains(a)

    # environment steps
    def _steps(self, x):
        for a in x:
            while 1:
                rnd = self.numx_rng.multivariate_normal(mdp.numx.zeros(self.action_dim), self._cov)
                self._a = self._m * self._a + (1 - self._m) * rnd
                if self._valid_action(self._a):
                    break
                self._a /= 2

            # explore/exploit
            f = self.numx_rng.rand() < self.epsilon
            a_ = f * self._a + (1 - f) * a

            phi, r, done, info = self.env.step(a_)
            self._render_step()
            if self.auto_reset and done:
                self.env.reset()
                self._a = mdp.numx.zeros(self.action_dim)
                self._epicnt += 1
            yield phi, a_, r, done, info

    def _train(self, x):
        self.epsilon *= self.decay ** x.shape[0]

    def reset(self):
        """Resets the environment, agent's position, the current iteration count and the epsiode count."""
        super(GymContinuousExplorerNode, self).reset()
        self.epsilon = self._init_epsilon
        self._a = mdp.numx.zeros(self.action_dim)
