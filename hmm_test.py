# coding: utf-8
import numpy as np
import random
import hmm
import time
import sys
# MA0466.1.txt trainning sequences file
# the motifs are located at the 51st nt
################### READ TFBS FILE #################
MOTIF_LEN = 11
TRUE_LOC = 50
N_TRAIN = 100
N_TEST = 500
MOTIF_FH = "./MA0466.1.txt"
PWM_FH = "./MA0466.1.pfm.txt"
PROMOTER_FH = "hgTables.txt"
with open(MOTIF_FH) as fh:
    lines = map(lambda l: l.strip(), fh.readlines())
all_seqs = []
for i in range(0,len(lines)/3):
    seq = ''.join(lines[i*3+1:i*3+3])
    all_seqs.append(seq.upper())
# keep the first 100 sequences for testing
test_seqs = all_seqs[:N_TEST]
# randomly choose 200 sequences for training
train_seqs = random.sample(all_seqs[N_TEST:], N_TRAIN)

################ INITIALIZE PARAMETERS ##############
# count dinucleotides frequency
fc = np.zeros((4,4))
dn = {0:'A',1:'C', 2:'G',3:'T'}
for seq in all_seqs:
    for i in range(4):
        for j in range(4):
            fc[i,j] += seq.count(dn[i]+dn[j])
fc = fc / (1.0 * np.sum(fc, axis=1)[:,None])
# construct TFBS transition mat
# fc = np.random.random_sample((4,4))
# fc = fc / np.sum(fc,axis=1)
tf_trans = np.zeros((4+MOTIF_LEN,4+MOTIF_LEN))
tf_trans[:4,:4] = fc / 1.25 # set Backgroud => Motif probabilities be 0.2
tf_trans[:4,4] = 0.2
# set the transition prob in Motif from i => i+1 be 1.0
np.fill_diagonal(tf_trans[4:4+MOTIF_LEN-1, 5:4+MOTIF_LEN], 1.0)
# set the Motif => Backgroud/Motf probabilites
tf_trans[4+MOTIF_LEN-1,:5] = np.array([0.23,0.23,0.23,0.23,0.08])
# construct the emission mat
tf_emit = np.zeros((4+MOTIF_LEN, 4))
# Background emit the corresponding nt with prob 1.0
np.fill_diagonal(tf_emit[:4,:4],1.0)
# load PWM
pwm = np.loadtxt(PWM_FH)
pwm = pwm / np.sum(pwm,axis=0)
tf_emit[4:,:] = pwm.T
# set start probabilites
tf_start = np.zeros(4+MOTIF_LEN)
tf_start[:5] = [0.23,0.23,0.23,0.23,0.08]
tf_symbol = ['A','C','G','T']
# set states, A,C,G,T for Backgroud, 1-11 for Motif
tf_states = ['A','C','G','T'] + map(lambda i: str(i), range(1, MOTIF_LEN+1))

################## INITIALIZE HMM #################
# init HMM
h_tf = hmm.HMM(tf_symbol, tf_states, tf_start, tf_trans, tf_emit)
# first test untrained HMM on testing data
v_untrained = []
for seq in test_seqs:
    v_path = h_tf.viterbi(seq)
    indices = [i for i, x in enumerate(v_path) if x == "1"]
    v_untrained.append(indices)

##################### TRAIN MHH ###################
# train hmm with trainning data
fh = open('bw_record.txt','a+')
start_time = time.time()
lls, diffs = h_tf.bw_train(train_seqs, update='st', max_iter=30)
fh.write("Training --- %s seconds --- \n" % (time.time() - start_time))
fh.close()

##################### TEST MHH ####################
if sys.argv[1].startswith("--"):
    if sys.argv[1][2:] == "test":
        # test hmm with viterbi on testing data
        start_time = time.time()
        v_trained = []
        for seq in test_seqs:
            v_path = h_tf.viterbi(seq)
            indices = [i for i, x in enumerate(v_path) if x == "1"]
            v_trained.append(indices)

        # calculate FDR
        def calculate_FDR(results, true_loc=50):
            n_true = 0
            n_false = 0
            for v_res in results:
                n_true += v_res.count(true_loc)
                n_false += len(v_res) - v_res.count(true_loc)
            return n_true, n_false
        # for untrained model
        n_true_untrain, n_false_untrain = calculate_FDR(v_untrained, TRUE_LOC)
        # for trained model
        n_true, n_false = calculate_FDR(v_trained, TRUE_LOC)
        fdr_untrain = n_false_untrain * 1.0 / (n_true_untrain + n_false_untrain)
        fdr_train = n_false * 1.0 / (n_true + n_false)

        fh = open('bw_record.txt','a+')
        fh.write("Testing on %d test sequences.\n" % N_TEST)
        fh.write("Untrained Model: True positive = %d, False positive = %d, FDR = %f \n" % (n_true_untrain, n_false_untrain, fdr_untrain))
        fh.write("Trained Model: True positive = %d, False positive = %d, FDR = %f \n" % (n_true, n_false, fdr_train))
        fh.write("Testing --- %s seconds --- \n" % (time.time() - start_time))
        fh.close()

    elif sys.argv[1][2:] == 'chr':
        from Bio import SeqIO
        ############## TEST ON REAL CHROMOSOME ############
        # read the chr20 promoters fasta file
        handle = open(PROMOTER_FH,'rU')
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        # record sequence and locatio of promoters
        chr20_promoters = []
        for record in records:
            ind_start = record.description.find(':')
            ind_end = record.description.find('-')
            start = int(record.description[ind_start+1:ind_end])
            seq = str(record.seq)
            name = record.name
            strand = record.description[record.description.find('strand=') + 7]
            chr20_promoters.append(((start, name, strand), seq))

        start_time = time.time()
        for info, seq in chr20_promoters:
            v_path = h_tf.viterbi(seq)
            with open('find_motif.txt', 'a+') as fhm:
                for i, x in enumerate(v_path):
                    if x == "1" and len(v_path) > i+10:
                        fhm.write(
                            'chr20' + "\t" + 
                            str(info[0]+i) + "\t" + 
                            str(info[0]+i+10) + "\t" + 
                            info[1] + "\t" +
                            info[2] + "\t" +
                            seq[i:i+11]+ "\n")

        fh = open('bw_record.txt','a+')
        fh.write("Testing --- %s seconds ---\n" % (time.time() - start_time))
        fh.close()

