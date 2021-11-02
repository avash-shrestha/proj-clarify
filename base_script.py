# This script assumes that a string using a word from human_context is TRUE and a string using a word from animal_context
# is false. The ambiguous pairs (human/nature and animal/urban) are tested for correctness on AI21 Jurassic-Jumbo.
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

# Trevor's api
# api_key = "KEArioXJKgpnEkhDLQbLBiGdfe0a8Knq"
# Avash's api
api_key = "nGAafrSAtOgan0kUVLepbhelY6HMeMJr"
NUM_QUERIES = 10
# thing is either human or animal, place is either urban or nature
Request = namedtuple("Request", ["thing", "place"])

# FUTURE: make list of namedtuples
template1 = ["The ", " is in a "]
template2 = ["The ", " is in an "]

human_context = {"person", "child", "man", "officer", "teacher", "salesperson", "politician", "chef", "artist"}
animal_context = {"tiger", "iguana", "toad", "butterfly", "wolf", "goat", "bat", "bear", "mosquito"}
urban_context = {"theater", "building", "city", "street", "shop", "school", "dwelling"}
nature_context = {"meadow", "river", "pond", "desert", "prairie", "jungle", "swamp"}

human_ambig = {"human", "toddler", "woman", "doctor", "firefighter", "soldier", "banker", "actor", "architect"}
animal_ambig = {"hawk", "elephant", "ant", "mouse", "crocodile", "shark", "sheep", "lion", "salamander"}
urban_ambig = {"skyscraper", "restaurant", "alley", "store", "apartment", "condominium", "house", "office"}
nature_ambig = {"ocean", "tundra", "forest", "cave", "canyon", "lake", "stream", "savannah"}


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
def multiple_context_requests(shots, alternate=True):
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # ambig_set is comprised of human/nature and animal/urban pairings from ambig sets.
    pick_set = set()
    pick_set.update(generate_x_y_requests(human_ambig, nature_ambig),
                    generate_x_y_requests(human_ambig, urban_ambig),
                    generate_x_y_requests(animal_ambig, nature_ambig),
                    generate_x_y_requests(animal_ambig, urban_ambig))
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open(str(shots) + "shots_responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = NUM_QUERIES
    completed_queries = 0

    # creates a 2(shots) + 1 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    usedLists = set()
    while completed_queries < num_queries:
        contextTrue = []
        for i in range(shots):
            ctxTrue = random.choice(tuple(human_urban_ctxt))
            while ctxTrue in contextTrue:
                ctxTrue = random.choice(tuple(human_urban_ctxt))
            contextTrue.append(ctxTrue)
        contextFalse = []
        for i in range(shots):
            ctxFalse = random.choice(tuple(animal_nature_ctxt))
            while ctxFalse in contextFalse:
                ctxFalse = random.choice(tuple(animal_nature_ctxt))
            contextFalse.append(ctxFalse)
        pick = random.choice(
            tuple(pick_set))  # All ambiguous/non-ambiguous combos, we figure out correctness in data.py
        if not alternate:
            totalContext = contextTrue + contextFalse
            random.shuffle(totalContext)
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
            usedLists.add((tuple(totalContext), pick))
            prompt = ""
            for i in range(len(totalContext)):
                if totalContext[i] in human_urban_ctxt:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                else:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: " 
            prompt = prompt.strip()

            # expect either TRUE or FALSE as the answer
            response = str(requests.post(
                "https://api.ai21.com/studio/v1/j1-jumbo/complete",
                headers={"Authorization": "Bearer " + api_key},
                json={
                    "prompt": prompt,
                    "numResults": 1,
                    "maxTokens": 1,
                    "stopSequences": ["."],
                    "topKReturn": 10,
                    "temperature": 0.0
                }
            ).json())
            if response.startswith("{'detail':"):
                continue
            data_file.write(response)
            data_file.write("\n")
            if response == "{'detail': 'Quota exceeded.'}":
                print(response)
                break
            completed_queries += 1
            time.sleep(3.1)

        else: #alternate = true
            if (contextTrue, contextFalse, pick) in usedLists:
                continue
            usedLists.add((contextTrue, contextFalse, pick))
            prompt = ""
            if random.choice([True, False]):  # Put Context True first
                for i in range(shots):
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                prompt += "A: "
            else:  # Put Context False first
                for i in range(shots):
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: " 
            prompt = prompt.strip()

            # expect either TRUE or FALSE as the answer
            response = str(requests.post(
                "https://api.ai21.com/studio/v1/j1-jumbo/complete",
                headers={"Authorization": "Bearer " + api_key},
                json={
                    "prompt": prompt,
                    "numResults": 1,
                    "maxTokens": 1,
                    "stopSequences": ["."],
                    "topKReturn": 10,
                    "temperature": 0.0
                }
            ).json())
            if response.startswith("{'detail':"):
                continue
            data_file.write(response)
            data_file.write("\n")
            if response == "{'detail': 'Quota exceeded.'}":
                print(response)
                break
            completed_queries += 1
            time.sleep(3.1)
    data_file.close()


multiple_context_requests(2, False)


def query_requests():
    # human_urban and animal_nature are the context sets
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # ambig_set is comprised of human/nature and animal/urban pairings from ambig sets.
    pick_set = set()
    pick_set.update(generate_x_y_requests(human_ambig, nature_ambig),
                    generate_x_y_requests(human_ambig, urban_ambig),
                    generate_x_y_requests(animal_ambig, nature_ambig),
                    generate_x_y_requests(animal_ambig, urban_ambig))
    used_set = set()
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = NUM_QUERIES
    completed_queries = 0

    # creates a 3 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    while completed_queries < num_queries:
        c1 = random.choice(tuple(human_urban_ctxt))  # Context True
        c2 = random.choice(tuple(animal_nature_ctxt))  # Context False
        pick = random.choice(
            tuple(pick_set))  # All ambiguous/non-ambiguous combos, we figure out correctness in data.py
        if (c1, c2, pick) in used_set:
            continue
        used_set.add((c1, c2, pick))
        if random.choice([True, False]):  # Put Context True first
            prompt = "Q: " + convert_to_sent(c1).strip() + "\r\n" + "A: TRUE" + "\r\n" + "Q: " + convert_to_sent(
                c2).strip() + "\r\n" + "A: FALSE" + "\r\n" + "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
        else:  # Put Context False first
            prompt = "Q: " + convert_to_sent(c2).strip() + "\r\n" + "A: FALSE" + "\r\n" + "Q: " + convert_to_sent(
                c1).strip() + "\r\n" + "A: TRUE" + "\r\n" + "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
        prompt = prompt.strip()
        # expect either TRUE or FALSE as the answer
        response = str(requests.post(
            "https://api.ai21.com/studio/v1/j1-jumbo/complete",
            headers={"Authorization": "Bearer " + api_key},
            json={
                "prompt": prompt,
                "numResults": 1,
                "maxTokens": 1,
                "stopSequences": ["."],
                "topKReturn": 10,
                "temperature": 0.0
            }
        ).json())
        if response.startswith("{'detail':"):
            continue
        data_file.write(response)
        data_file.write("\n")
        if response == "{'detail': 'Quota exceeded.'}":
            break
        completed_queries += 1
        time.sleep(3.1)
    data_file.close()


# Return true if the sentence contains a human, false if it contains an animal
def query_ambig_requests():
    # human_urban and animal_nature are the context sets
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # ambig_set is comprised of human/nature and animal/urban pairings from ambig sets.
    ambig_set = set()
    ambig_set.update(generate_x_y_requests(human_ambig, nature_ambig), generate_x_y_requests(animal_ambig, urban_ambig))
    used_set = set()
    actual_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    data_file = open("responses_ambig_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = NUM_QUERIES
    completed_queries = 0

    # creates a 3 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    while completed_queries < num_queries:
        c1 = random.choice(tuple(human_urban_ctxt))
        c2 = random.choice(tuple(animal_nature_ctxt))
        ambig = random.choice(tuple(ambig_set))
        if (c1, c2, ambig) in used_set:
            continue
        used_set.add((c1, c2, ambig))
        prompt = "Q: " + convert_to_sent(c1).strip() + "\r\n" + "A: TRUE" + "\r\n" + "Q: " + convert_to_sent(
            c2).strip() + "\r\n" + "A: FALSE" + "\r\n" + "Q: " + convert_to_sent(ambig).strip() + "\r\n" + "A: "
        prompt = prompt.strip()
        # expect either TRUE or FALSE as the answer
        response = str(requests.post(
            "https://api.ai21.com/studio/v1/j1-jumbo/complete",
            headers={"Authorization": "Bearer " + api_key},
            json={
                "prompt": prompt,
                "numResults": 1,
                "maxTokens": 1,
                "stopSequences": ["."],
                "topKReturn": 10,
                "temperature": 0.0
            }
        ).json())
        data_file.write(str((c1, c2, ambig)))
        data_file.write("\n")
        data_file.write(response)
        data_file.write("\n")
        if response == "{'detail': 'Quota exceeded.'}":
            break
        completed_queries += 1
        time.sleep(3.1)
    data_file.close()


# Return true if the sentence contains a human, false if it contains an animal
def query_non_ambig_requests():
    # human_urban and animal_nature are the context sets
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # non_ambig_set is comprised of human/urban and animal/nature pairings from ambig sets.
    non_ambig_set = set()
    non_ambig_set.update(generate_x_y_requests(human_ambig, urban_ambig),
                         generate_x_y_requests(animal_ambig, nature_ambig))
    used_set = set()
    actual_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    data_file = open("responses_non_ambig_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = NUM_QUERIES
    completed_queries = 0

    # creates a 3 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    while completed_queries < num_queries:
        c1 = random.choice(tuple(human_urban_ctxt))
        c2 = random.choice(tuple(animal_nature_ctxt))
        non_ambig = random.choice(tuple(non_ambig_set))
        if (c1, c2, non_ambig) in used_set:
            continue
        used_set.add((c1, c2, non_ambig))
        prompt = "Q: " + convert_to_sent(c1).strip() + "\r\n" + "A: TRUE" + "\r\n" + "Q: " + convert_to_sent(
            c2).strip() + "\r\n" + "A: FALSE" + "\r\n" + "Q: " + convert_to_sent(non_ambig).strip() + "\r\n" + "A: "
        prompt = prompt.strip()
        # expect either TRUE or FALSE as the answer
        response = str(requests.post(
            "https://api.ai21.com/studio/v1/j1-jumbo/complete",
            headers={"Authorization": "Bearer " + api_key},
            json={
                "prompt": prompt,
                "numResults": 1,
                "maxTokens": 1,
                "stopSequences": ["."],
                "topKReturn": 10,
                "temperature": 0.0
            }
        ).json())
        data_file.write(str((c1, c2, non_ambig)))
        data_file.write("\n")
        data_file.write(response)
        data_file.write("\n")
        if response == "{'detail': 'Quota exceeded.'}":
            break
        completed_queries += 1
        time.sleep(3.1)
    data_file.close()

# query_requests()

# query_ambig_requests()
# query_non_ambig_requests()
