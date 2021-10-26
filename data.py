import ast
import json
import math
import pickle
import random
import time
from time import gmtime, strftime
from collections import namedtuple
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import base_script as bs
import os

#FILENAME = "responses_2021-10-25T_03-09-31Z.txt"


def analyze_generic():
    for filename in os.listdir("responses"): 
        with open("responses\\" + filename, "r", encoding="utf8") as f:
            data = f.readlines()
            tmp_set = set()
            datadict = {"is_ambig" : [], "is_correct" : [], "is_human" : [], "is_context_true_first": [], "percent_certainty_of_correct" : []}
            for line in data:
                is_correct = False
                is_ambig = False
                is_human = False
                is_context_true_first = False
                line = ast.literal_eval(line)
                completion = line["completions"][0]["data"]["text"].strip()
                prompt = line["prompt"]["text"].split()
                # 2 is first thing, 6 is first place, 8 is first answer
                # 11 is second thing, 15 is second place, 17 is second answer
                # 20 is third thing, 24 is third place
                #first check if ambig or not by comparing third thing and thrid place
                    #if ambig20 is in human and ambig24 is nature or ambig20 is in animal and ambig24 is in urban
                if prompt[20] in bs.human_ambig and prompt[24] in bs.nature_ambig or prompt[20] in bs.animal_ambig and prompt[24] in bs.urban_ambig: 
                    is_ambig = True
                if prompt[20] in bs.human_ambig and completion == "TRUE" or prompt[20] in bs.animal_ambig and completion == "FALSE": 
                    is_correct = True
                if prompt[20] in bs.human_ambig: 
                    is_human = True
                if prompt[8] == "TRUE": 
                    is_context_true_first = True

                first_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["token"].strip()
                second_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["token"].strip()
                tmp_set.add(first_token)
                tmp_set.add(second_token)
                first_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["logprob"])
                second_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["logprob"])
                normalizing_prob = first_prob + second_prob  # since there is small percentage of answer that isn't T/F
                percent_certain = first_prob / normalizing_prob if is_correct else second_prob / normalizing_prob

                datadict["is_ambig"].append(is_ambig)
                datadict["is_correct"].append(is_correct)
                datadict["is_human"].append(is_human)
                datadict["is_context_true_first"].append(is_context_true_first)
                datadict["percent_certainty_of_correct"].append(percent_certain)

            df = pd.DataFrame(datadict)
            outf = open("dataframe_files\dataframe__" + filename + ".csv", "wb")
            df.to_csv(outf)
            outf.close()
        f.close()

    




            
'''
        print(tmp_set)


def analyze():
    with open(FILENAME, "r", encoding="utf8") as f:
        data = f.readlines()
        human_corr, human_not_corr, human_tot, avg_human_corr_prob = 0.0, 0.0, 0.0, 0.0
        animal_corr, animal_not_corr, animal_tot, avg_animal_corr_prob = 0.0, 0.0, 0.0, 0.0
        avg_prob = 0.0
        tmp_set = set()
        for line in data:
            if line[0] == '(':
                continue
            line = ast.literal_eval(line)
            completion = line["completions"][0]["data"]["text"].strip()
            prompt = line["prompt"]["text"].split()
            first_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["token"].strip()
            second_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["token"].strip()
            tmp_set.add(first_token)
            tmp_set.add(second_token)
            first_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["logprob"])
            second_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["logprob"])
            normalizing_prob = first_prob + second_prob
            avg_prob += normalizing_prob
            if prompt[20] in base_human_ambig:  # want completion to be TRUE
                human_tot += 1
                if completion == "TRUE":
                    avg_human_corr_prob += first_prob / normalizing_prob
                    human_corr += 1
                else:
                    avg_human_corr_prob += second_prob / normalizing_prob
                    human_not_corr += 1
            else:  # prompt[20] in animal_ambig  # want completion to be FALSE
                animal_tot += 1
                if completion == "FALSE":
                    avg_animal_corr_prob += first_prob / normalizing_prob
                    animal_corr += 1
                else:
                    avg_animal_corr_prob += second_prob / normalizing_prob
                    animal_not_corr += 1
        print("HUMAN: ", human_corr, human_not_corr, human_tot, human_corr / human_tot)
        print("ANIMAL: ", animal_corr, animal_not_corr, animal_tot, animal_corr / animal_tot)
        print("PROBS: ", avg_human_corr_prob / human_tot, avg_animal_corr_prob / animal_tot,
              avg_prob / (human_tot + animal_tot))
        print(tmp_set)


<<<<<<< HEAD
analyze_generic()
'''
analyze_generic()
=======
analyze_generic() '''
>>>>>>> 771367af84f069e5dc13f2d05a862905544e0ac0
