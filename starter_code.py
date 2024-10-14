# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:32:48 2018

@author: jdkan
"""
import time
import alignment
import copy
from itertools import product
# Your task is to *accurately* predict the primer melting points using machine 
# learning based on the sequence of the primer.

# Load the primers and their melting points.


def CalculatePrimerFeatures(seq):
    return [len(seq), seq.count('A'), seq.count('C'), seq.count('T'),seq.count('G'),seq.count('GC')] # .95

    # return [len(seq)] + [seq.count(''.join(chars)) for i in range(1,4) for chars in product("ACTG",repeat=i)] # slower but .96


def GetMeltingPoint(primer:str,melting_point_rf):
    return melting_point_rf.predict([CalculatePrimerFeatures(primer)])
    
# template sequence is 5' -> 3'
# primer is 3' -> 5'

def get_smallest_index_match_on_template_sequence(template_sequence:str, primer:str):
    scoring = alignment.ScoreParam(10,-5,-7)
    # binding if at least 80% max score
    SCORE_THRESHOLD = .8 * (scoring.match*len(primer))
    best, optloc, matrix = alignment.local_align(template_sequence, primer, score=scoring)
    if best < SCORE_THRESHOLD:
        return 9999999

    return optloc[0]-1 # the more straight-forward approach

    for i in range(len(template_sequence)+1):
        for j in range(len(primer),-1,-1):
            if(matrix[i][j]>=SCORE_THRESHOLD):
                print(i,j,matrix[i][j])
                return max(0,i-1) # includes primer
    return 9999999 # MAX_INT
            

def reverse_complement(primer:str):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    comp_match = ''.join(complement[base] for base in reversed(primer))
    return comp_match

def complement(primer:str):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    comp_match = ''.join(complement[base] for base in primer)
    return comp_match

def PredictPCRProduct(primer1, primer2, template_sequence, melting_point_rf):
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
        print("Lengths not right!",len(primer1),len(primer2))
        return None
        
    #Fail any primers with conflicting melting points 
    melting_point_1, melting_point_2 = GetMeltingPoint(primer1,melting_point_rf), GetMeltingPoint(primer2,melting_point_rf)
    if abs(melting_point_1-60)>2 or abs(melting_point_2-60)>2:
        print("temps not right",melting_point_1,melting_point_2)
        return None
        
    #Check for binding on both primers
    top_strand_match = min(
        get_smallest_index_match_on_template_sequence(template_sequence = template_sequence, primer = reverse_complement(primer1)),
        get_smallest_index_match_on_template_sequence(template_sequence = template_sequence, primer = reverse_complement(primer2)))
    
    bottom_strand_match = len(template_sequence)-1-(
        min(get_smallest_index_match_on_template_sequence(template_sequence = reverse_complement(template_sequence), primer = reverse_complement(primer1)),
            get_smallest_index_match_on_template_sequence(template_sequence = reverse_complement(template_sequence), primer = reverse_complement(primer2)))) # reverse the primers

    if top_strand_match-bottom_strand_match<=0 or bottom_strand_match<0 or top_strand_match > 99999:
        return None # no overlap
    
    

    return template_sequence[bottom_strand_match:top_strand_match+1]

def LoadFastA(path):
    infile = open(path, 'r')
    seq = ""
    infile.readline()
    for line in infile:
        seq += line[:-1]
    return seq
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   from sklearn.metrics import r2_score
   from sklearn.ensemble import RandomForestRegressor
   import importlib
   import os
   import sys

   from Bio.Seq import Seq
   from sklearn.ensemble import RandomForestRegressor

   print("Running Task 1:")
   
   infile = open("training_primers.txt", 'r')
   infile.readline() # don't load headers
   primers = []
   melting_points = []
   features = []
   st = time.time()
   for line in infile:
       Line = line.split()
       primers.append(Line[0])
       melting_points.append(float(Line[1]))
       # calculate features
       features.append(CalculatePrimerFeatures(Line[0]))
   feat_time = (time.time()-st)/(len(features)/1000)
   # cross validation
   how_many_folds = 10 
   predictions = []
   truth = []
   my_len = len(features[-1])
    
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



   
   """
   Task 2:
   Design a function to predict whether a product will be made in a PCR reaction.
   Your function should take as input the template DNA and the two primers and 
   return the product or 'None'.

   This requires a local alignment function which is provided for you or you 
   can use another implementation.
    
   There are test cases in PCR_product_test_cases.txt.
   
   """
   task2_randomforest = RandomForestRegressor(n_estimators = 200)
   task2_randomforest.fit(features, melting_points)
   
#    print(PredictPCRProduct("ACTG", "ACTG", "ACTCAGCGACTGC", task2_randomforest))
#    print(PredictPCRProduct("TGGTGGGATGTCTTTCAACAGG", "AACTACGGAGAACTACAGCAACCT","ACGTCAGCGAGCGCTACGACGTGGTGGGATGTCTTTCAACAGGACGGACTGACGCGACGACTGACTGTAGGCTAGGTTGCTGTAGTTCTCCGTAGTTAGCTACGACGCATGCAGCTGCA", task2_randomforest))
   assert PredictPCRProduct("TGGTGGGATGTCTTTCAACAGG", "AACTACGGAGAACTACAGCAACCT","ACGTCAGCGAGCGCTACGACGTGGTGGGATGTCTTTCAACAGGACGGACTGACGCGACGACTGACTGTAGGCTAGGTTGCTGTAGTTCTCCGTAGTTAGCTACGACGCATGCAGCTGCA", task2_randomforest) == "TGGTGGGATGTCTTTCAACAGGACGGACTGACGCGACGACTGACTGTAGGCTAGGTTGCTGTAGTTCTCCGTAGTT"

   
   """
   Task 3:
   Design primers for a PCR reaction to distinguish between the three (strands)
   types of DNA.  
   
   -Your primers should be between 18 and 35 bases long.  
   -They should have at least 80% match to the DNA strand.
   -Predicted melting points of any primers to be run in the same reaction
   should be between 58.0 and 62.0 C.
   -Products are distinguishable in length if their difference in length is >40
   bases
   -Your products should not be longer than 1000 bases.
   
   We are making predictions about the functionality of sets of primers.  We 
   will synthesize your group's primers and test them in the lab later.
     
   """

def generatePrimers(sequences):
    """
        Given 3 sequences, select 3 different unique reverse primer starting
        locations that meet the conditions listed below, and are of length 30.
        The forward strand will start at the same location on all 3 strands.
    """

    import random
    deliverable = []
    for seq in sequences:
        while True: 
            #Generate a random (reverse complment)
            n = len(seq)
            rand_n = random.randint(30, n - 30)
            s_1 = reverse_complement(seq[0:30])
            s_2 = complement(seq[rand_n: rand_n + 30])
            
            # make sure that they satisfy melting point criterion
            melting_point, melting_point_2 = GetMeltingPoint(s_1, task2_randomforest)
            #  = GetMeltingPoint(s_2, task2_randomforest)

            # and they don't bind to other strands
            counter = 0
            for s in sequences:
                if s == seq: 
                    # we want the primers to bind to seq.
                    pass
                # Use PCR product w/candidate primers: makes sure no product 
                # will not be generated
                if (PredictPCRProduct(s_1,s_2,s,melting_point) is None):
                    counter == 1
                    
                """
                for i in range(len(s) - 30):
                    if complement(s_1) == s[i:i+30] or complement(s_2) == s[i:i+30]:
                        counter = 1
                """
            if (melting_point <= 62 and melting_point >= 58 and 
               melting_point_2 <= 62 and melting_point_2 >= 58 and counter == 0):
                deliverable.append(s_1, s_2)
                break
        
    return deliverable

test = ["TGCAGTCGCTCGCGATGCTGCACCACCCTACAGAAAGTTGTCCTGCCTGACTGCGCTGCTGACTGACATCCGATCCAACGACATCAAGAGGCATCAATCGATGCTGCGTACGTCGACGT", "TGCAGTCGCTCGCGATGCTGCACCACCCTACAGAAAGTTGTCCTGCCTGACTGCGCTGCTGACTGACATCCGATCCAACGACATCAAGAGGCATCAATCGATGCTGCGTACGTCGACGT", 
"TGCAGTCGCTCGCGATGCTGCACCACCCTACAGAAAGTTGTCCTGCCTGACTGCGCTGCTGACTGACATCCGATCCAACGACATCAAGAGGCATCAATCGATGCTGCGTACGTCGACGT"]
print(generatePrimers(test))

   
   
# R^2 - how good it is compared to a flat line .8 okay

# TGCAGTCGCTCGCGATGCTGCACCACCCTACAGAAAGTTGTCCTGCCTGACTGCGCTGCTGACTGACATCCGATCCAACGACATCAAGAGGCATCAATCGATGCTGCGTACGTCGACGT AGGTTGCTGTAGTTCTCCGTAGTT

# TGCAGTCGCTCGCGATGCTGCACCACCCTACAGAAAGTTGTCCTGCCTGACTGCGCTGCTGACTGACATCCGATCCAACGACATCAAGAGGCATCAATCGATGCTGCGTACGTCGACGT GGACAACTTTCTGTAGGGTGGT
# TGCAGTCGCTCGCGATGCTGCACCACCCTACAGAAAGTTGTCCTGCCTGACTGCGCTGCTGACTGACATCCGATCCAACGACATCAAGAGGCATCAATCGATGCTGCGTACGTCGACGT TCCAACGACATCAAGAGGCATCAA