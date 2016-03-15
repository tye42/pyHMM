states = ['Fair','Unfair']
symbols = range(1,7)
trans = np.array([[0.60,0.40], [0.20, 0.80]])
emit = np.array([[1.0 / 6 for i in range(6)], [0.1,0.1,0.1,0.1,0.1,0.5]])
start = np.array([0.5,0.5])
end = np.array([0.5,0.5])
h = hmm.HMM(symbols, states, start, trans, emit)