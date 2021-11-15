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
import openai

# OPENAI API
api_key = "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"
NUM_QUERIES = 1
# thing is either human or animal, place is either urban or nature
Request = namedtuple("Request", ["thing", "place"])

# FUTURE: make list of namedtuples
template1 = ["The ", " is in a "]
template2 = ["The ", " is in an "]

human_context = {"person", "child", "man", "officer", "teacher", "salesperson", "politician", "chef", "artist", "builder", "dancer", "athlete"}
animal_context = {"tiger", "iguana", "toad", "butterfly", "wolf", "goat", "bat", "bear", "mosquito", "horse", "meerkat", "dolphin"} # "owl", "squirrel", "spider", "moose"}
urban_context = {"theater", "building", "city", "street", "shop", "school", "dwelling", "factory",  "garage", "courthouse", "hotel", "warehouse"}
nature_context = {"meadow", "river", "pond", "desert", "prairie", "jungle", "swamp", "sea", "rainforest", "taiga", "grassland", "bay"}

human_ambig = {"human", "toddler", "woman", "doctor", "firefighter", "soldier", "banker", "actor", "architect", "butcher", "engineer", "student"}
animal_ambig = {"hawk", "elephant", "ant", "mouse", "crocodile", "shark", "sheep", "lion", "salamander", "bee", "condor", "chipmunk"}# "buffalo", "panda"}
urban_ambig = {"skyscraper", "restaurant", "alley", "store", "apartment", "condominium", "house", "office", "museum", "casino", "hospital", "library"}# "airport"}
nature_ambig = {"ocean", "tundra", "forest", "cave", "canyon", "lake", "stream", "savannah", "stream", "creek", "delta", "valley" }


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


# Return true if the sentence contains a human, false if it contains an animal
def multiple_context_requests(shots, order, ambig=True, add_disambig=False):
    encoding = 'utf-8'
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # ambig_set is comprised of human/nature and animal/urban pairings from ambig sets.
    pick_set = set()
    if ambig:
        pick_set.update(generate_x_y_requests(human_ambig, nature_ambig),
                        generate_x_y_requests(animal_ambig, urban_ambig))
    else:
        pick_set.update(generate_x_y_requests(human_ambig, urban_ambig),
                        generate_x_y_requests(animal_ambig, nature_ambig))
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("OPENAI_" + str(shots) + "shots_" +
                     "order_" + str(order) + "_" +
                     ("ambig_" if ambig else "NOTambig_") +
                     ("disambig_" if add_disambig else "NONEdisambig_") +
                     "responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = NUM_QUERIES
    completed_queries = 0
    # creates a 2(shots) + 1 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    usedLists = set()
    while completed_queries < num_queries:
        contextTrue = []
        if add_disambig:
            tmp_shots = shots - 1
        else:
            tmp_shots = shots
        for i in range(tmp_shots):
            ctxTrue = random.choice(tuple(human_urban_ctxt))
            while ctxTrue in contextTrue:
                ctxTrue = random.choice(tuple(human_urban_ctxt))
            contextTrue.append(ctxTrue)
        contextFalse = []
        for i in range(tmp_shots):
            ctxFalse = random.choice(tuple(animal_nature_ctxt))
            while ctxFalse in contextFalse:
                ctxFalse = random.choice(tuple(animal_nature_ctxt))
            contextFalse.append(ctxFalse)
        pick = random.choice(
            tuple(pick_set))  # All ambiguous/non-ambiguous combos, we figure out correctness in data.py
        if order == 2:
            totalContext = contextTrue + contextFalse
            random.shuffle(totalContext)
            if add_disambig:
                if random.choice([True, False]):
                    # add human disambig first
                    totalContext.append(
                        Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))))
                    totalContext.append(
                        Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context))))
                else:
                    # add animal disambig first
                    totalContext.append(
                        Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context))))
                    totalContext.append(
                        Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))))
            # check TF alternating pattern
            altCheck = True
            for i in range(len(totalContext)):
                if i % 2 == 0 and totalContext[i] not in human_urban_ctxt:
                    altCheck = False
                    break
                if i % 2 != 0 and totalContext[i] not in animal_nature_ctxt:
                    altCheck = False
                    break
            if altCheck:
                continue
            # check FT alternating pattern
            altCheck = True
            for i in range(len(totalContext)):
                if i % 2 == 0 and totalContext[i] not in animal_nature_ctxt:
                    altCheck = False
                    break
                if i % 2 != 0 and totalContext[i] not in human_urban_ctxt:
                    altCheck = False
                    break
            if altCheck:
                continue
            print(totalContext, pick)
            usedLists.add((tuple(totalContext), pick))
            prompt = ""
            for i in range(len(totalContext)):
                if totalContext[i][0] in human_context:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                else:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"

            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
            prompt = prompt.strip()

            # expect either TRUE or FALSE as the answer
            response = requests.get(
                "https://api.openai.com/v1/engines/davinci/completions/browser_stream",
                headers={"Authorization": "Bearer " + api_key},
                stream=True,
                params={
                    "prompt": prompt,
                    "n": 1,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 2
                })
            whole = ""
            for line in response:
                rite = str(line, encoding)
                whole += rite
            whole = whole.strip().removesuffix("data: [DONE]").strip().removeprefix("data: ")
            prompt_lst = prompt.split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots):
                prompt_dict["context"].append(
                    ((Request(prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.')), prompt_lst[8 + i * 18]),
                     (Request(prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.')), prompt_lst[17 + i * 18])))
            prompt_dict["query"].append(Request(prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            data_file.write(str(prompt_dict))
            data_file.write("\n")
            data_file.write(whole)
            data_file.write("\n")
            completed_queries += 1
            # time.sleep(3.1)

        else:  # alternate = true
            if (add_disambig):
                pairDisambig = (Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))),
                                Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context))))
                if (tuple(contextTrue), tuple(contextFalse), pairDisambig, pick) in usedLists:
                    continue
                usedLists.add((tuple(contextTrue), tuple(contextFalse), pairDisambig, pick))
            else:
                if (tuple(contextTrue), tuple(contextFalse), pick) in usedLists:
                    continue
                usedLists.add((tuple(contextTrue), tuple(contextFalse), pick))
            prompt = ""
            if order == 1:  # Put Context True first
                for i in range(tmp_shots):
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                if (add_disambig):
                    prompt += "Q: " + convert_to_sent(pairDisambig[0]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(pairDisambig[1]).strip() + "\r\n" + "A: FALSE" + "\r\n"
            else:  # Put Context False first
                for i in range(tmp_shots):
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                if (add_disambig):
                    prompt += "Q: " + convert_to_sent(pairDisambig[1]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(pairDisambig[0]).strip() + "\r\n" + "A: TRUE" + "\r\n"

            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
            prompt = prompt.strip()
            # expect either TRUE or FALSE as the answer
            response = requests.get(
                "https://api.openai.com/v1/engines/davinci/completions/browser_stream",
                headers={"Authorization": "Bearer " + api_key},
                stream=True,
                params={
                    "prompt": prompt,
                    "n": 1,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 2
                })
            whole = ""
            for line in response:
                rite = str(line, encoding)
                whole += rite
            whole = whole.strip().removesuffix("data: [DONE]").strip().removeprefix("data: ")
            prompt_lst = prompt.split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots):
                prompt_dict["context"].append(
                    ((Request(prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.')), prompt_lst[8 + i * 18]),
                     (Request(prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.')), prompt_lst[17 + i * 18])))
            prompt_dict["query"].append(Request(prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            data_file.write(str(prompt_dict))
            data_file.write("\n")
            data_file.write(whole)
            data_file.write("\n")
            completed_queries += 1
            # time.sleep(3.1)

    data_file.close()


# multiple_context_requests(1, 2)
# multiple_context_requests(2, 1, True, True)
# multiple_context_requests(2, 0, True, True)
# multiple_context_requests(3, 2)
# multiple_context_requests(4, 2)
# multiple_context_requests(5, 2)




