import numpy as np


class HMM(object):
    """
    Class for hidden markov model. 
    """

    def __init__(self, symbols, states, start_prob, transition_mat, emission_mat):
        self.symbols = symbols
        self.states = states
        self.n_symbols = len(symbols)
        self.n_states = len(states)
        self.start_prob = start_prob
        self.transition_mat = transition_mat  # row (state) -> col (state)
        self.emission_mat = emission_mat  # row (state) - > col (symbol)
        # map state => index
        self._state_dict = {x: i for i, x in enumerate(states)}
        # map symbol => index
        self._symbol_dict = {x: i for i, x in enumerate(symbols)}
        # self._start_prob_log = np.log(start_prob)

    def sample(self, length):
        observations = []
        state_path = []
        curr_state = np.random.choice(self.n_states, p=self.start_prob)
        for dummy_i in range(length):
            emmit_symbol = np.random.choice(
                self.n_symbols, p=self.emission_mat[curr_state, :].flatten())
            observations.append(self.symbols[emmit_symbol])
            state_path.append(self.states[curr_state])
            curr_state = np.random.choice(
                self.n_states, p=self.transition_mat[curr_state, :].flatten())
        return observations, state_path

    def _eexp(self, x):
        return 0 if np.isnan(x) else np.exp(x)

    def _eln(self, x):
        if x == 0:
            return np.nan
        elif x > 0:
            return np.log(x)
        else:
            raise ValueError('Negative input error')

    def _elnsum(self, x, y):
        if np.isnan(x) or np.isnan(y):
            if np.isnan(x):
                return y
            else:
                return x
        else:
            if x > y:
                return x + self._eln(1 + np.exp(y - x))
            else:
                return y + self._eln(1 + np.exp(x - y))

    def _elnproduct(self, x, y):
        if np.isnan(x) or np.isnan(y):
            return np.nan
        else:
            return x + y

    def _forward(self, observations):
        # implement the log-space forward algorithm
        n_observations = len(observations)
        # convert the sequence of symbols to a sequence of index from
        # 0..|symbols|-1
        obs_ids = map(lambda s: self._symbol_dict[s], observations)
        f_mat = np.zeros((self.n_states, n_observations))
        # init the forward matrix with log(a_0,t * e_t[y[0]])
        for i in range(self.n_states):
            f_mat[i, 0] = self._elnproduct(
                self._eln(self.emission_mat[i, obs_ids[0]]), self._eln(self.start_prob[i]))
        for t in range(1, n_observations):
            for j in range(self.n_states):
                logalpha = np.nan
                for i in range(self.n_states):
                    logalpha = self._elnsum(logalpha, self._elnproduct(
                        f_mat[i, t - 1], self._eln(self.transition_mat[i, j])))
                f_mat[j, t] = self._elnproduct(
                    logalpha, self._eln(self.emission_mat[j, obs_ids[t]]))
        return f_mat

    def _backward(self, observations):
        n_observations = len(observations)
        obs_ids = map(lambda s: self._symbol_dict[s], observations)
        b_mat = np.zeros((self.n_states, n_observations))
        b_mat[:, n_observations - 1] = 0
        for t in range(n_observations - 2, -1, -1):
            for i in range(self.n_states):
                logbeta = np.nan
                for j in range(self.n_states):
                    # correct an error in emission prob, in paper, without eln
                    logbeta = self._elnsum(
                        logbeta, self._elnproduct(self._eln(self.transition_mat[i, j]), self._elnproduct(self._eln(self.emission_mat[j, obs_ids[t + 1]]), b_mat[j, t + 1])))
                b_mat[i, t] = logbeta
        return b_mat

    def _prob_at_state(self, f_mat, b_mat):
        """
        Calculate P(pi_i = s | y_j) = f_s(i)b_s(i) / P(y_j)
        for all states s and time point i in log space
        """
        g_mat = np.zeros((f_mat.shape))
        for t in range(f_mat.shape[1]):
            normalizer = np.nan
            for i in range(self.n_states):
                g_mat[i, t] = self._elnproduct(f_mat[i, t], b_mat[i, t])
                normalizer = self._elnsum(normalizer, g_mat[i, t])
            for i in range(self.n_states):
                g_mat[i, t] = self._elnproduct(g_mat[i, t], -normalizer)
        return g_mat

    def _expect_state(self, f_mat, b_mat, observations):
        e_mat = np.full((self.n_states, self.n_states), np.nan)
        for t in range(len(observations) - 1):
            normalizer = np.nan
            for i in range(self.n_states):
                for j in range(self.n_states):



    #def train(self, data, max_iter=100, delta=1e-9):
