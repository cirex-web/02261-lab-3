# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:32:48 2018

@author: jdkan
"""
import alignment
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import random

# Your task is to *accurately* predict the primer melting points using machine 
# learning based on the sequence of the primer.

# Load the primers and their melting points.


def CalculatePrimerFeatures(seq):
    return [len(seq), seq.count('A'), seq.count('C'), seq.count('T'),seq.count('G'),seq.count('GC')] # R^2 = 0.9505965920164953
    # return [len(seq)] + [seq.count(''.join(chars)) for i in range(1,4) for chars in product("ACTG",repeat=i)] # slower but .96


def GetMeltingPoint(primer:str,melting_point_rf):
    return melting_point_rf.predict([CalculatePrimerFeatures(primer)])
    
# template sequence is 5' -> 3'
# primer is 3' -> 5'

def get_smallest_index_match_on_template_sequence(template_sequence:str, primer:str):
    scoring = alignment.ScoreParam(10,-5,-7)
    # binding if at least 80% max score
    SCORE_THRESHOLD = .9 * (scoring.match*len(primer))
    best, optloc, matrix = alignment.local_align(template_sequence, primer, score=scoring)
    # if best < SCORE_THRESHOLD:
    #     return 9999999
    # return optloc[0]-1+1 # the more straight-forward approach

    for i in range(len(template_sequence)+1):
        for j in range(len(primer),-1,-1):
            if(matrix[i][j]>=SCORE_THRESHOLD):
                # print(i,j,matrix[i][j])
                return i # includes primer
    return 9999999 # MAX_INT
            

def reverse_complement(primer:str):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C','N':'N'}
    comp_match = ''.join(complement[base] for base in reversed(primer))
    return comp_match

def reverse(primer:str):
    return primer[-1::-1]

def complement(primer:str):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    comp_match = ''.join(complement[base] for base in primer)
    return comp_match

def PredictPCRProduct(primer1, primer2, template_sequence, melting_point_rf, skip_temp = False):
    """
    Input:
        primer1 = a primer sequence in 5' to 3' order
        primer2 = a primer sequence in 5' to 3' order
        template_sequence = sequence from which we are trying to generate 
        copies using PCR in 5' to 3' order.  Assume this is double stranded, 
        but we are only including the top strand in the argument.
        melting_point_rf = random forest learned from task1 to predict primer
        melting points.
    Output:
        return sequence of successful PCR amplication reaction or None (if 
        there is no successful reaction)
        
    """
    #Fail any primers with length out of range 18-30 bp
    if(not ((18 <= len(primer1) <= 30) and (18 <= len(primer2) <= 30))):
        # print("Lengths not right!",len(primer1),len(primer2))
        return None
        
    #Fail any primers with conflicting melting points 
    if not skip_temp:
        melting_point_1, melting_point_2 = GetMeltingPoint(primer1,melting_point_rf), GetMeltingPoint(primer2,melting_point_rf)
        if (abs(melting_point_1-60)>2 or abs(melting_point_2-60)>2):
            # print("temps not right",melting_point_1,melting_point_2)
            return None
            
    #Check for binding on both primers
    top_strand_match = min(
        get_smallest_index_match_on_template_sequence(template_sequence = template_sequence, primer = reverse_complement(primer1)),
        get_smallest_index_match_on_template_sequence(template_sequence = template_sequence, primer = reverse_complement(primer2)))
    
    bottom_strand_match = len(template_sequence)-1-(
        min(get_smallest_index_match_on_template_sequence(template_sequence = reverse_complement(template_sequence), primer = reverse_complement(primer1)),
            get_smallest_index_match_on_template_sequence(template_sequence = reverse_complement(template_sequence), primer = reverse_complement(primer2)))) # reverse the primers
    # print(top_strand_match,bottom_strand_match)
    if top_strand_match>=bottom_strand_match or bottom_strand_match<0 or top_strand_match > 99999 or bottom_strand_match-top_strand_match>1000:
        return None # no overlap
    

    return template_sequence[top_strand_match:bottom_strand_match+1]
actual_seqs = ["tcccggatgttagcggcggacgggtgagtaacacgtgggtaacctgcctgtaagactgggataactccgggaaaccggagctaataccggatagttccttgaaccgcatggttcaaggatgaaagacggtttcggctgtcacttacagatggacccgcggcgcattagctagttggtgaggtaacggctcaccaaggcgacgatgcgtagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttccgcaatggacgaaagtctgacggagcaacgccgcgtgagtgatgaaggttttcggatcgtaaagctctgttgttagggaagaacaagtgcaagagtaactgcttgcaccttgacggtacctaaccagaaagccacggctaactacgtgccagcagccgcggtaatacgtaggtggcaagcgttgtccggAattattgggcgtAaagggctcgcaggcggtttcttaagtCtgatgtgaaagcccccggctcaaccggggagggtcattggaaactgggaaacttgagtgcagaagaggagagtggaattccacgtgtagcggtgaaatgcgtagagatgtggaggaacaccagtggcgaaggcgactctctggtctgtaactgacgctgaggagcgaaagcgtggggagcgaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaagtgttagggggtttccgccccttagtgctgcagctaacgcattaagcactccgcctggggagtacggtcgcaagactgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcctctgacaaccctagagatagggctttcccttcggggacagagtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttgatcttagttgccagcattcagttgggcactctaaggtgactgccggtgacaaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggacagaacaaagggctgcgagaccgcaaggtttagccaatcccacaaatctgttctcagttcggatcgcagtctgcaactcgactgcgtgaagctggaatcgctagtaatcgcggatcagcatgccgcggtgaatacgttcccgggccttgtacacaccgcccgtcacaccacgagagtttgcaacacccgaagtcggtgaggtaactttatggagccagcc","ggcgtgcctaatacatgcaagtcgagcggatcgatgggagcttgcnncntgagatcagcggcggacgggtgagtaacacgtgggtaacctgcctgtaagactgggataactccgggaaaccggggctaataccggataacacctacccccgcatgggggaaggttgaaaggtggcttcggctatcacttacagatggacccgcggcgcattagctagttggtgaggtaatggctcaccaaggcgacgatgcgtagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttccgcaatggacgaaagtctgacggagcaacgccgcgtgagtgaagaaggttttcggatcgtaaaactctgttgttagggaagaacaagtgccgttcgaatagggcggcgccttgacggtacctaaccagaaagccacggctaactacgtgccagcagccgcggtaatacgtaggtggcaagcgttgtccggaattattgggcgtaaagcgcgcgcaggtggtttcttaagtctgatgtgaaagcccacggctcaaccgtggagggtcattggaaactggggaacttgagtgcagaagaggaaagtggaattccaagtgtagcggtgaaatgcgtagatatttggaggaacaccagtggcgaaggcgactttctggtctgtaactgacactgaggcgcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaagtgttagagggtttccgccctttagtgctgcagctaacgcattaagcactccgcctggggagtacggtcgcaagactgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcctctgacaaccctagagatagggctttccccttcgggggacagagtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttgatcttagttgccagcattcagttgggcactctaagatgactgccggtgacaaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggacggtacaaagggctgcaagaccgcgaggtttagccaatcccataaaaccgttctcagttcggattgtaggctgcaactcgcctacatgaagctggaatcgctagtaatcgcgnatcagcatgccgcggtgaatacgttcccgggccttgtacacaccgcccgtcacaccacgagagtttgtaaca"]
actual_seqs = [seq.upper() for seq in actual_seqs]


def task_3(sequences,rf):
    def test_primer_validity(forward_primer,backward_primer):
        product_lens = []
        for sequence in sequences:
            product = PredictPCRProduct(forward_primer,backward_primer,sequence,rf,skip_temp=False)
            if product is None:
                return False
            product_lens.append(len(product))

        return True

    ref_seq = sequences[0]
    primer_length = 23
    while True:
        i = random.randint(0,len(ref_seq)-primer_length-1)

        # pick a random start index for the reverse primer (orientation 5' -> 3'), making sure it comes after the forward primer so we get some overlap
        j = random.randint(0,max(0,len(ref_seq)-1-(i+primer_length-1)-primer_length-1))

        fp = ref_seq[i:primer_length+i]
        bp = reverse_complement(ref_seq)[j:primer_length+j]

        fp = reverse_complement(fp)
        bp = reverse_complement(bp)
        assert len(fp)==len(bp)==primer_length
        if(test_primer_validity(fp,bp)):
            print(fp,bp,i,j) # result

# Note: I wrote this to only work on two sequences cuz I was lazy, but the same randomized procedure should work on more
def task_5(sequences,rf):
    ref_seq = sequences[0]
    primer_length = 23 # fix our primer length at 23 for ALL generated primers
    while True:
        i = random.randint(0,len(ref_seq)-primer_length-1) # randomly picked start index for universal forward primer (relative to the first sequence)
        j = random.randint(0,max(0,len(ref_seq)-1-(i+primer_length-1)-primer_length-1)) # randomly picked start index for reverse primer for first sequence
        k = random.randint(0,max(0,len(sequences[1])-1-(i+primer_length-1)-primer_length-1)) # randomly picked start index for reverse primer for second sequence

        # note that all generated primers have length 23
        fp = ref_seq[i:primer_length+i]
        bp1 = reverse_complement(ref_seq)[j:primer_length+j]
        bp2 = reverse_complement(sequences[1])[k:primer_length+k]

        fp = reverse_complement(fp)
        bp1 = reverse_complement(bp1)
        bp2 = reverse_complement(bp2)
        
        assert len(fp)==len(bp1)==len(bp2)==primer_length
        # ensure that only the intended forward and reverse primers made a product of unique length
        product_1 = PredictPCRProduct(fp,bp1,sequences[0],rf,skip_temp=False)
        not_product_1 = PredictPCRProduct(fp,bp2,sequences[0],rf,skip_temp=False) or PredictPCRProduct(bp1,bp2,sequences[0],rf,skip_temp=False)
        product_2 = PredictPCRProduct(fp,bp2,sequences[1],rf,skip_temp=False)
        not_product_2 = PredictPCRProduct(fp,bp1,sequences[1],rf,skip_temp=False) or PredictPCRProduct(bp1,bp2,sequences[1],rf,skip_temp=False)

        if product_1 is not None and product_2 is not None and not not_product_1 and not not_product_2 and abs(len(product_1)-len(product_2))>=100:
            print(abs(len(product_1)-len(product_2)))
            print(product_1,product_2,fp,bp1,bp2) # result

def LoadFastA(path):
    infile = open(path, 'r')
    seq = ""
    infile.readline()
    for line in infile:
        seq += line[:-1]
    return seq

def set_up_testing_data():
    infile = open("training_primers.txt", 'r')
    infile.readline() # don't load headers

    primers = []
    melting_points = []
    features = []

    for line in infile:
       Line = line.split()
       primers.append(Line[0])
       melting_points.append(float(Line[1]))
       features.append(CalculatePrimerFeatures(Line[0]))

    return features, melting_points

def cross_validate(features, melting_points):
    # cross validation
    how_many_folds = 10 
    predictions = []
    truth = []

    for fold in range(how_many_folds):
        #print ("Calculating Fold",fold)
        training_features = []
        training_outcomes = []
        testing_features = []
        testing_outcomes = []
        for c in range(len(melting_points)):
            if c % how_many_folds == fold:
                # put this one in testing data
                testing_features.append(features[c])
                testing_outcomes.append(melting_points[c])
            else:
                # put this one in training data
                training_features.append(features[c])
                training_outcomes.append(melting_points[c])
        # train the model
        
        rf = RandomForestRegressor(n_estimators = 200)
        rf.fit(training_features, training_outcomes)
        fold_predictions = rf.predict(testing_features)
        truth += testing_outcomes
        predictions += list(fold_predictions)
        
            
    #truth = np.array(truth)
    #predictions = np.array(predictions)        
    print("Task 1 Results:\n")
    print("R2 Score:", r2_score(truth, predictions))


if __name__ == "__main__":


    features, melting_points = set_up_testing_data()
    # cross_validate(features,melting_points)


    task2_randomforest = RandomForestRegressor(n_estimators = 200)
    task2_randomforest.fit(features, melting_points)

    # task_3(actual_seqs,task2_randomforest)

    # task_5(actual_seqs,task2_randomforest)