# HMM Motif Finder

[Project site](https://tye42.github.io/2016/03/17/Hidden-Markov-Model-Motif-Finder.html)

Python library for Hidden Markov models with Baum-Welch algorithm. 

Documentation

```
hmm.HMM(symbols, states, start_prob, transition_mat, emission_mat)
    HMM class initialization.
    Parameters:
        symbols: list, emission symbols
        states: list, hidden states
        start_prob: numpy.ndarray, start probability to each state
        transition_mat: numpy.ndarray, transition matrix between each pair of states
        emission_mat: numpy.ndarray, emission matrix for a state emit a symbol

hmm.sample(length, size=1)
    Sampling a give size of observations using HMM.
    Parameters:
        length: int, the length of a simulated observations sequence
        size: int, how many sequences to simulated, default 1
    Returns:
        samples: a list of list contains simulated sequences
        states: a list of list contains true states path

hmm.loglikelihood(data):
    Calculate the loglikelihood of give set of observations.
    Parameters:
        data: list of sequences of observations.
    Returns:
        the loglikelihood of all sequences

hmm.bw_train(data, update='ste', max_iter=100, d=1e-9):
    Run Baum-Welch algorithm on a give set of training data.
    Parameters:
        data: a set of training data
        update: indicate which parameter to update
        max_iter: the maximum iterations
        d: a threshod to check whether we can stop
    Returns:
        logll_all_iter: the loglikelihoods for each iteration
        diff_all_iter: the difference of the update parameters and old parameters

hmm.viterbi(obs):
    Run viterbi algorithm to find the most likely states path.
    Parameters:
        obs: a give sequence of observations
    Returns:
        v_path: a list of most likely states path
```
