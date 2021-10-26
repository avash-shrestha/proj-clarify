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

human_context = {"person", "child", "man", "officer", "teacher", "salesperson", "politician", "chef", "artist"}
animal_context = {"tiger", "iguana", "toad", "butterfly", "wolf", "goat", "bat", "bear", "mosquito"}
urban_context = {"theater", "building", "city", "street", "shop", "school", "dwelling"}
nature_context = {"meadow", "river", "pond", "desert", "prairie", "jungle", "swamp"}

human_ambig = {"human", "toddler", "woman", "doctor", "firefighter", "soldier", "banker", "actor", "architect"}
animal_ambig = {"hawk", "elephant", "ant", "mouse", "crocodile", "shark", "sheep", "lion", "salamander"}
urban_ambig = {"skyscraper", "restaurant", "alley", "store", "apartment", "condominium", "house", "office"}
nature_ambig = {"ocean", "tundra", "forest", "cave", "canyon", "lake", "stream", "savannah"}

FILENAME = "responses_2021-10-24T_19-46-36Z.txt"


def analyze_generic():
    with open(FILENAME, "r", encoding="utf8") as f:
        data = f.readlines()
        """
        stats:
            ambig:
                human:
                    TF:
                    FT:
                animal:
                    TF:
                    FT:
            non-ambig:
                human:
                    TF:
                    FT:
                animal:
                    TF:
                    FT:
        """

        tmp_set = set()
        for line in data:
            line = ast.literal_eval(line)
            completion = line["completions"][0]["data"]["text"].strip()
            prompt = line["prompt"]["text"].split()
            # 2 is first thing, 6 is first place, 8 is first answer
            # 11 is second thing, 15 is second place, 17 is second answer
            # 20 is third thing, 24 is third place
            first_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["token"].strip()
            second_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["token"].strip()
            tmp_set.add(first_token).add(second_token)
            first_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["logprob"])
            second_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["logprob"])
            normalizing_prob = first_prob + second_prob  # since there is small percentage of answer that isn't T/F
            if prompt[20] in human_ambig and prompt[24].strip(".") in nature_ambig or \
                    prompt[20] in animal_ambig and prompt[24].strip(".") in urban_ambig:
                if prompt[20] in human_ambig:  # want to be TRUE
                    if first_token == "TRUE":

                    else:
                        ds
                else:  # want to be FALSE
            else:  # non-ambig
                ds


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
            if prompt[20] in human_ambig:  # want completion to be TRUE
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


analyze_generic()
