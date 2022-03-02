import ast
import json
import math
import pickle
import random
import time
from time import gmtime, strftime
from collections import namedtuple
from datetime import date, datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
NUM_QUERIES = 200
# thing is either human or animal, place is either urban or nature
Request = namedtuple("Request", ["thing", "place"])

# FUTURE: make list of namedtuples
template1 = ["The ", " is in a "]
template2 = ["The ", " is in an "]

human_context = {"person", "child", "man", "officer", "teacher", "salesperson", "politician", "chef", "artist",
                 "builder", "dancer", "athlete"}
animal_context = {"tiger", "iguana", "toad", "butterfly", "wolf", "goat", "bat", "bear", "mosquito", "horse", "meerkat",
                  "dolphin"}  # "owl", "squirrel", "spider", "moose"}
urban_context = {"theater", "building", "city", "street", "shop", "school", "dwelling", "factory", "garage",
                 "courthouse", "hotel", "warehouse"}
nature_context = {"meadow", "river", "pond", "desert", "prairie", "jungle", "swamp", "sea", "rainforest", "taiga",
                  "grassland", "bay"}

human_query = {"human", "toddler", "woman", "doctor", "firefighter", "soldier", "banker", "actor", "architect",
               "butcher", "engineer", "student"}
animal_query = {"hawk", "elephant", "ant", "mouse", "crocodile", "shark", "sheep", "lion", "salamander", "bee",
                "condor", "chipmunk"}  # "buffalo", "panda"}
urban_query = {"skyscraper", "restaurant", "alley", "store", "apartment", "condominium", "house", "office", "museum",
               "casino", "hospital", "library"}  # "airport"}
nature_query = {"ocean", "tundra", "forest", "cave", "canyon", "lake", "stream", "savannah", "stream", "creek", "delta",
                "valley"}


def generate_x_y_requests(x, y):
    request_set = set()
    for first in x:
        for second in y:
            request_set.add(Request(first, second))
    return request_set


def convert_to_sent(r):
    """
    Makes and returns an English sentence from the pair of words passed in, r.
    """
    template = template2 if r.place[0].lower() in "aeiuo" else template1
    return template[0] + r.thing + template[1] + r.place + "."


def random_learning(shots):
    # for the starting context
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # for the rest of the query
    CONST_MISMATCH, CONST_MATCH = set(), set()
    CONST_MISMATCH.update(generate_x_y_requests(human_query, nature_query),
                          generate_x_y_requests(animal_query, urban_query))
    CONST_MATCH.update(generate_x_y_requests(human_query, urban_query),
                       generate_x_y_requests(animal_query, nature_query))
    num_run = 20
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("RANDOMLearning_OPENAI_" + str(shots) + "shots_" + "responses_" + str(num_run) + "_runs_" + actual_time + ".txt", 'w',
                     encoding="utf-8")
    # begin iterations
    current_run = 0
    while current_run < num_run:
        # start with 1 context pair
        # 50/50 chance of True/False or False/True initial context
        if random.choice([0, 1]) == 0:
            context = [random.choice(list(human_urban_ctxt)), random.choice(list(animal_nature_ctxt))]
        else:
            context = [random.choice(list(animal_nature_ctxt)), random.choice(list(human_urban_ctxt))]
        # reset the pick sets
        match_pick_set, mismatch_pick_set = set(CONST_MATCH), set(CONST_MISMATCH)
        curr_shot = 1
        while curr_shot <= shots:
            # create 4 match examples
            examples = []
            for i in range(4):
                examples.append(match_pick_set.pop())
            # create 1 mismatch examples
            examples.append(mismatch_pick_set.pop())
            context.append(random.choice(examples))
            curr_shot += 1
        # store context pairs and match/mismatch info in data file
        complete_dict = {"context_info": [], "match_type_info": []}
        context_prompt = ""
        for i, ctxt in enumerate(context):
            complete_dict["context_info"].append(tuple(ctxt))
            # if ctxt is a match pair
            if ctxt in CONST_MATCH or ctxt in human_urban_ctxt or ctxt in animal_nature_ctxt:
                complete_dict["match_type_info"].append(True)
            else:
                # if ctxt is a mismatch pair
                complete_dict["match_type_info"].append(False)
            # format context and examples to presentable tokens
            # skip the answer for the last Q/A
            if i != (len(context) - 1):
                context_prompt += "Q: " + convert_to_sent(ctxt).strip() + "\r\n" + "A: " + ("TRUE" if (ctxt.thing in human_context or ctxt.thing in human_query) else "FALSE") + "\r\n"
            else:
                context_prompt += "Q: " + convert_to_sent(ctxt).strip() + "\r\n" + "A: "
        prompt = context_prompt
        prompt = prompt.strip()
        # call API
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            n=1,
            logprobs=2,
            echo=True,
        )
        # CHECK THIS BEFORE DOING BIG RUNS
        cleaned_str = str(response["choices"]).replace("\n", "").replace("      ", " ").replace("    ", " ") \
            .replace("   ", " ").replace("  ", " ").replace("{ ", "{").replace(" }", "}").replace("[ ", "[") \
            .replace(" ]", "]").partition(": ")[2].removesuffix("]")
        data_file.write(cleaned_str)
        data_file.write("\n")
        str_complete_dict = str(complete_dict).replace("\'", "\"")
        data_file.write(str_complete_dict)
        data_file.write("\n")
        # go next iteration
        current_run += 1


# random_learning(8)

def active_learning(shots):
    # for the starting context
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # for the rest of the query
    # complete sets, used to reset the pick sets every run later
    CONST_MISMATCH, CONST_MATCH = set(), set()
    CONST_MISMATCH.update(generate_x_y_requests(human_query, nature_query),
                          generate_x_y_requests(animal_query, urban_query))
    CONST_MATCH.update(generate_x_y_requests(human_query, urban_query),
                       generate_x_y_requests(animal_query, nature_query))
    num_run = 2
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("ActiveLearning_OPENAI_" + str(shots) + "shots_" + "responses_" + str(num_run) + "_runs_" + actual_time + ".txt", 'w',
                     encoding="utf-8")
    # begin iterations
    curr_run = 0
    while curr_run < num_run:
        curr_shot = 1
        # start with 1 context pair
        # 50/50 chance of True/False or False/True initial context
        if random.choice([0, 1]) == 0:
            context = [random.choice(list(human_urban_ctxt)), random.choice(list(animal_nature_ctxt))]
        else:
            context = [random.choice(list(animal_nature_ctxt)), random.choice(list(human_urban_ctxt))]
        # reset the pick sets
        match_pick_set, mismatch_pick_set = set(CONST_MATCH), set(CONST_MISMATCH)
        while curr_shot <= shots:
            # store probabilities to find smallest abs difference later
            examples_probs = {}
            # store THING/PLACE pairs and raw probs to put into datafile later for easier data collection
            info_dict = {}
            # store everything for easier data collection
            complete_dict = {}
            examples = []
            # create 4 match examples
            for i in range(4):
                examples.append(match_pick_set.pop())
            # create 1 mismatch examples
            examples.append(mismatch_pick_set.pop())
            context_prompt = ""
            # only shuffle the last n - 2 things in context, leave the first two things in initial order
            context_to_shuffle = context[2:]
            # make context prompt randomly
            random_context = context[:2] + random.sample(context_to_shuffle, len(context_to_shuffle))
            # add info about context
            complete_dict["context_info"] = []
            # format context and examples to presentable tokens
            for ctxt in random_context:
                complete_dict["context_info"].append(tuple(ctxt))
                context_prompt += "Q: " + convert_to_sent(ctxt).strip() + "\r\n" + "A: " + \
                                  ("TRUE" if (ctxt.thing in human_context or ctxt.thing in human_query) else "FALSE") + "\r\n"
            # go through each example
            for example in examples:
                prompt = context_prompt
                prompt += "Q: " + convert_to_sent(example).strip() + "\r\n" + "A: "
                prompt = prompt.strip()
                # now lets say response works
                # expect either TRUE or FALSE as the answer
                response = openai.Completion.create(
                    engine="davinci",
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    n=1,
                    logprobs=2,
                    echo=True,
                )
                # CHECK THIS BEFORE DOING BIG RUNS
                cleaned_str = str(response["choices"]).replace("\n", "").replace("      ", " ").replace("    ", " ") \
                    .replace("   ", " ").replace("  ", " ").replace("{ ", "{").replace(" }", "}").replace("[ ", "[") \
                    .replace(" ]", "]").partition(": ")[2].removesuffix("]")
                data_file.write(cleaned_str)
                data_file.write("\n")
                cleaned_dct = json.loads(cleaned_str)
                # collect the probabilities of the returned token being " FALSE" and " TRUE"
                raw_F, raw_T = math.exp(cleaned_dct["logprobs"]["top_logprobs"][-1][" FALSE"]), math.exp(
                    cleaned_dct["logprobs"]["top_logprobs"][-1][" TRUE"])
                # normalize them
                norm_F, norm_T = raw_F / (raw_F + raw_T), raw_T / (raw_F + raw_T)
                # absolute difference of normalized probabilities
                examples_probs[example] = abs(norm_T - norm_F)
                # add pairs and probs to info_dict for later: [raw_F, raw_T, isMatch]
                info_dict[tuple(example)] = [raw_F, raw_T, True if example in CONST_MATCH else False]
            complete_dict["query_info"] = info_dict
            # find true smallest example (smallest difference in probabilities)
            smallest_example = examples[0]
            smallest_prob = examples_probs[smallest_example]
            for example in examples_probs:
                if smallest_prob > examples_probs[example]:
                    smallest_example = example
                    smallest_prob = examples_probs[smallest_example]
            if smallest_example in CONST_MATCH:
                # if smallest_example is a match pair
                complete_dict["picked_example"] = (tuple(smallest_example), True)
            else:
                # if smallest_example is a mismatch pair
                complete_dict["picked_example"] = (tuple(smallest_example), False)
            # add complete_dict to data_file
            data_file.write(str(complete_dict))
            data_file.write("\n")
            # add to context, "Active Learning"
            context.append(smallest_example)
            # go next iteration
            curr_shot += 1
        curr_run += 1

active_learning(3)

# NOTES **********************************************************************************************
# create 4 matched and 1 mismatched
# ambig --> mismatched
# disambig --> matched
# create 4 ambig examples
# MISMATCHED --> False
# Matched --> True
# for isMatch
# we want it to pick the 1 mismatched example and not the 4 matched examples