import numpy as np
from scipy.misc import logsumexp


class HMM(object):
    """
    Class for hidden markov model.
    Methods:
        sample(length, size): generate a list of simulated observations, each with give length
        loglikelihood(data): return the loglikelihood of a set of data
        bw_train(data, update, max_iter, d): train HMM with data using Baum-Welch
        viterbi(obs): return the most possible states path using viterbi
    """

    def __init__(self, symbols, states, start_prob, transition_mat, emission_mat):
        """
        Args:
            symbols: list, emission symbols
            states: list, hidden states
            start_prob: numpy.ndarray, start probability to each state
            transition_mat: numpy.ndarray, transition matrix between each pair of states
            emission_mat: numpy.ndarray, emission matrix for a state emit a symbol
        """
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

    def _check(self):
        pass

    def _check_symbol(self, obs):
        pass

    def sample(self, length, size=1):
        """
        Sampling a give size of observations using HMM.
        Args:
            length: int, the length of a simulated observations sequence
            size: int, how many sequences to simulated, default 1
        Returns:
            samples: a list of list contains simulated sequences
            states: a list of list contains true states path
        """
        samples = []
        states = []
        for dummy_k in range(size):
            obs = []
            state_path = []
            # init state
            curr_state = np.random.choice(self.n_states, p=self.start_prob)
            for dummy_i in range(length):
                emmit_symbol = np.random.choice(
                    self.n_symbols, p=self.emission_mat[curr_state, :].flatten())
                obs.append(self.symbols[emmit_symbol])
                # transit to next state
                state_path.append(self.states[curr_state])
                curr_state = np.random.choice(
                    self.n_states, p=self.transition_mat[curr_state, :].flatten())
            samples.append(obs)
            states.append(state_path)
        return samples, states

    def _forward(self, obs):
        """
        Calculate the forward matrix.
        Args:
            obs: a list or string of observations
        Returns:
            f_mat: numpy.ndarray, |states| * |observations|
        """
        # implement the log-space forward algorithm
        n_obs = len(obs)
        # convert the sequence of symbols to a sequence of index from
        # 0..|symbols|-1
        obs_ids = map(lambda s: self._symbol_dict[s], obs)
        f_mat = np.full((self.n_states, n_obs), -np.inf)
        # init the forward matrix with log(a_0,t * e_t[y[0]])
        for i in range(self.n_states):
            if self.start_prob[i] > 0 and self.emission_mat[i, obs_ids[0]] > 0: # log(0)
                f_mat[i, 0] = np.log(self.start_prob[i]) + np.log(self.emission_mat[i, obs_ids[0]])
            else:
                f_mat[i, 0] = -np.inf
        for t in range(1, n_obs):
            for j in range(self.n_states):
                if self.emission_mat[j, obs_ids[t]] > 0:
                    logalpha = -np.inf
                    for i in range(self.n_states):
                        if self.transition_mat[i, j] > 0: # log(0)
                            tmp_alpha = f_mat[i, t - 1] + np.log(self.transition_mat[i, j])
                            if tmp_alpha > -np.inf:
                                logalpha = np.logaddexp(logalpha, tmp_alpha)
                    f_mat[j, t] = np.log(self.emission_mat[j, obs_ids[t]]) + logalpha
                else:
                    f_mat[j, t] = -np.inf
        return f_mat

    def _backward(self, obs):
        """
        Calculate the backward matrix.
        Args:
            obs: a list or string of observations
        Returns:
            b_mat: numpy.ndarray, |states| * |observations|
        """
        n_obs = len(obs)
        obs_ids = map(lambda s: self._symbol_dict[s], obs)
        b_mat = np.full((self.n_states, n_obs), -np.inf)
        # init last column
        b_mat[:, n_obs - 1] = 0
        for t in range(n_obs - 2, -1, -1):
            for i in range(self.n_states):
                logbeta = -np.inf
                for j in range(self.n_states):
                    if self.transition_mat[i, j] > 0 and self.emission_mat[j, obs_ids[t + 1]] > 0:
                        tmp_beta = b_mat[j, t + 1] + np.log(self.transition_mat[i, j]) + np.log(self.emission_mat[j, obs_ids[t + 1]])
                        if tmp_beta > -np.inf:
                            logbeta = np.logaddexp(logbeta, tmp_beta)
                b_mat[i, t] = logbeta
        return b_mat
    
    def loglikelihood(self, data):
        """
        Calculate the loglikelihood of give set of observations.
        Args:
            data: list of sequences of observations.
        Returns:
            the loglikelihood of all sequences
        """
        prob_data = []
        for obs in data:
            f_mat = self._forward(obs)
            prob_obs = logsumexp(f_mat[:,len(obs)-1])
            prob_data.append(prob_obs)
        return logsumexp(prob_data)

    def _bw_pass(self, obs, update):
        """
        Calculate expectaions of emissions, transition and start probabilites for a sequence.
        Args:
            obs: a sequence of observations
            update: any combination of 's', 't', 'e', to update start, transition, emission
        Returns:
            start_mat, trans_mat, emit_mat: matrix for start, transition and emission expectiation
            prob_obs: loglikelihood of observations
        """
        n_obs = len(obs)
        obs_ids = map(lambda s: self._symbol_dict[s], obs)
        start_mat = np.zeros(self.start_prob.shape)
        trans_mat = np.zeros(self.transition_mat.shape)
        emit_mat = np.zeros(self.emission_mat.shape)
        f_mat = self._forward(obs)
        b_mat = self._backward(obs)
        # calculate the log probability of observations P(y)
        # which is the sum of the last column of forward matrix
        prob_obs = -np.inf
        for i in range(self.n_states):
            tmp = f_mat[i, n_obs-1]
            if tmp > -np.inf:
                prob_obs = np.logaddexp(prob_obs, tmp)

        # calculate expectation of A_s,t
        if 't' in update:
            for i in range(self.n_states):
                for j in range(self.n_states):
                    tmp_sum = -np.inf
                    for t in range(n_obs-1):
                        if self.transition_mat[i,j] > 0 and self.emission_mat[j, obs_ids[t+1]] > 0:
                            tmp = f_mat[i, t] + np.log(self.transition_mat[i,j]) + np.log(self.emission_mat[j, obs_ids[t+1]]) + b_mat[j, t+1]
                            if tmp > -np.inf:
                                tmp_sum = np.logaddexp(tmp_sum, tmp)
                    trans_mat[i,j] = tmp_sum - prob_obs
        # calculate expectation of E_s(b)
        if 'e' in update:
            for i in range(self.n_states):
                for k in range(self.n_symbols):
                    tmp_sum = -np.inf
                    for t in range(n_obs):
                        if k == obs_ids[t]:
                            tmp = f_mat[i, t] + b_mat[i, t]
                            if tmp > -np.inf:
                                tmp_sum = np.logaddexp(tmp_sum, tmp)
                    emit_mat[i, k] = tmp_sum - prob_obs
        # calculate expectation start state
        # A_0,t = P(pi_1 = t | y) = f_t(1) * b_t(1) / P(y)
        if 's' in update:
            for i in range(self.n_states):
                start_mat[i] = f_mat[i, 0] + b_mat[i, 0] - prob_obs

        return start_mat, trans_mat, emit_mat, prob_obs

    def bw_train(self, data, update='ste', max_iter=100, d=1e-9):
        """
        Run Baum-Welch algorithm on a give set of training data.
        Args:
            data: a set of training data
            update: indicate which parameter to update
            max_iter: the maximum iterations
            d: a threshod to check whether we can stop
        Returns:
            logll_all_iter: the loglikelihoods for each iteration
            diff_all_iter: the difference of the update parameters and old parameters
        """
        diff_all_iter = []
        logll_all_iter = []
        for curr_iter in range(max_iter):
            start = np.full(self.start_prob.shape, -np.inf)
            trans = np.full(self.transition_mat.shape, -np.inf)
            emit = np.full(self.emission_mat.shape, -np.inf)
            prob_data = []
            for obs in data:
                tmp_start, tmp_trans, tmp_emit, prob_obs = self._bw_pass(obs, update)
                # sum expectation of start for all sequences
                prob_data.append(prob_obs)
                if 's' in update:
                    for i in range(self.n_states):
                        if tmp_start[i] > -np.inf:
                            start[i] = np.logaddexp(start[i], tmp_start[i])
                # sum expectation of transition for all sequences
                if 't' in update:
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            if tmp_trans[i,j] > -np.inf:
                                trans[i,j] = np.logaddexp(trans[i,j], tmp_trans[i,j])
                # sum expectation of emission for all sequences
                if 'e' in update:
                    for i in range(self.n_states):
                        for k in range(self.n_symbols):
                            if tmp_emit[i, k] > -np.inf:
                                emit[i, k] = np.logaddexp(emit[i, k], tmp_emit[i,k])

            # update start, tansition and emission probabilites
            logll_all_iter.append(logsumexp(prob_data))
            diff = 0
            if 's' in update:
                start = np.exp(start)
                start = start / (1.0 * np.sum(start))
                diff += np.sqrt(np.square(start - self.start_prob).sum())
                self.start_prob = start
            if 't' in update:
                trans = np.exp(trans)
                trans = trans / (1.0 * np.sum(trans, axis=1))[:, None]
                diff += np.sqrt(np.square(trans - self.transition_mat).sum())
                self.transition_mat = trans
            if 'e' in update:
                emit = np.exp(emit)
                emit = emit / (1.0 * np.sum(emit, axis=1))[:, None]
                diff += np.sqrt(np.square(emit - self.emission_mat).sum())
                self.emission_mat = emit
            diff_all_iter.append(diff)
            with open('bw_record.txt', 'a+') as fh:
                fh.write("Iteration %d: loglikelihood=%f, difference=%e \n" % (curr_iter+1, logll_all_iter[curr_iter], diff))
            if diff < d:
                break
        logll_all_iter.append(self.loglikelihood(data))
        return logll_all_iter, diff_all_iter

    def viterbi(self, obs):
        """
        Run viterbi algorithm to find the most likely states path.
        Args:
            obs: a give sequence of observations
        Returns:
            v_path: a list of most likely states path
        """
        n_obs = len(obs)
        obs_ids = map(lambda s: self._symbol_dict[s], obs)
        v_mat = np.full((self.n_states, n_obs), -np.inf)

        for i in range(self.n_states):
            if self.start_prob[i] > 0 and self.emission_mat[i, obs_ids[0]] > 0: # log(0)
                v_mat[i, 0] = np.log(self.start_prob[i]) + np.log(self.emission_mat[i, obs_ids[0]])
            else:
                v_mat[i, 0] = -np.inf
        for t in range(1, n_obs):
            for j in range(self.n_states):
                if self.emission_mat[j, obs_ids[t]] > 0:
                    max_v = -np.inf
                    for i in range(self.n_states):
                        if self.transition_mat[i, j] > 0: # log(0)
                            tmp_v = v_mat[i, t - 1] + np.log(self.transition_mat[i, j])
                            if tmp_v > max_v:
                                max_v = tmp_v
                    v_mat[j, t] = np.log(self.emission_mat[j, obs_ids[t]]) + max_v
                else:
                    v_mat[j, t] = -np.inf
        # trackback
        v_path = np.full(n_obs, None)
        v_path[n_obs-1] = np.argmax(v_mat[:,n_obs-1])
        for t in range(n_obs-2, -1, -1):
            for i in range(self.n_states):
                v_path[t] = np.argmax(
                    v_mat[:, t] + np.ma.log(self.transition_mat[:, v_path[t+1]]).filled(-np.inf))
        v_path = map(lambda s: self.states[int(s)], v_path)
        return v_path
