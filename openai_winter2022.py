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
    num_run = 50
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    # data_file = open("RANDOMLearning_OPENAI_" + str(shots) + "shots_" + "responses_" + str(num_run) + "_runs_" + actual_time + ".txt", 'w',
    #                  encoding="utf-8")
    data_file = open("sample5.txt", "w", encoding="utf-8")
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
                context_prompt += "Q: " + convert_to_sent(ctxt).strip() + "\n" + "A: " + (
                    "TRUE" if (ctxt.thing in human_context or ctxt.thing in human_query) else "FALSE") + "\n"
            else:
                context_prompt += "Q: " + convert_to_sent(ctxt).strip() + "\n" + "A: "
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


# random_learning(1)

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
    num_run = 20
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("ActiveLearning_OPENAI_" + str(shots) + "shots_" + "responses_" + str(
        num_run) + "_runs_" + actual_time + ".txt", 'w',
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
                                  ("TRUE" if (
                                              ctxt.thing in human_context or ctxt.thing in human_query) else "FALSE") + "\r\n"
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


# active_learning(8)

# NOTES **********************************************************************************************
# create 4 matched and 1 mismatched
# ambig --> mismatched
# disambig --> matched
# create 4 ambig examples
# MISMATCHED --> False
# Matched --> True
# for isMatch
# we want it to pick the 1 mismatched example and not the 4 matched examples
def multiple_context_requests(shots, order, ambig=True, num_disambig=0):
    encoding = 'utf-8'
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # ambig_set is comprised of human/nature and animal/urban pairings from ambig sets.
    pick_set = set()
    if ambig:
        pick_set.update(generate_x_y_requests(human_query, nature_query),
                        generate_x_y_requests(animal_query, urban_query))
    else:
        pick_set.update(generate_x_y_requests(human_query, urban_query),
                        generate_x_y_requests(animal_query, nature_query))
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("sanitycheck" + str(shots) + "shots_" +
                     "order_" + str(order) + "_" +
                     ("ambig_" if ambig else "NOTambig_") +
                     ("disambig_" + str(num_disambig) + "_" if num_disambig > 0 else "NONEdisambig_") +
                     "responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = 200
    completed_queries = 0
    # creates a 2(shots) + 1 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    usedLists = set()
    while completed_queries < num_queries:
        contextTrue = []
        tmp_shots = shots - num_disambig
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
            for i in range(num_disambig):
                # add human disambig first
                totalContext.append(
                    Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))))
                totalContext.append(
                    Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context))))
            # check TF alternating pattern
            random.shuffle(totalContext)
            altCheck = True
            if shots != 1:
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
                if totalContext[i][0] in human_context:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                else:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"

            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
            prompt = prompt.strip()

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
            prompt_lst = prompt.split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots):
                prompt_dict["context"].append(
                    (prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.'), prompt_lst[8 + i * 18]))
                prompt_dict["context"].append(
                    (prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.'), prompt_lst[17 + i * 18]))
            prompt_dict["query"].append((prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            cleaned_str = str(response["choices"]).replace("\n", "").replace("      ", " ").replace("    ", " ") \
                .replace("   ", " ").replace("  ", " ").replace("{ ", "{").replace(" }", "}").replace("[ ", "[") \
                .replace(" ]", "]").partition(": ")[2].removesuffix("]")
            data_file.write(cleaned_str)
            data_file.write("\n")
            completed_queries += 1
            # time.sleep(3.1)

        else:  # alternate = true
            if (num_disambig > 0):
                print("not possible")
                disambig_list = []
                while (tuple(contextTrue), tuple(contextFalse), tuple(disambig_list), pick) in usedLists:
                    for i in range(num_disambig):
                        disambig_list.append(
                            (Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))),
                             Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context)))))
                if (tuple(contextTrue), tuple(contextFalse), disambig_list, pick) in usedLists:
                    continue
                usedLists.add((tuple(contextTrue), tuple(contextFalse), disambig_list, pick))
            else:
                if (tuple(contextTrue), tuple(contextFalse), pick) in usedLists:
                    continue
                usedLists.add((tuple(contextTrue), tuple(contextFalse), pick))
            prompt = ""
            if order == 1:  # Put Context True first
                for i in range(tmp_shots):
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                if (num_disambig > 0):
                    print("T/F NOT POSSIBLE")
                    for i in range(len(disambig_list)):
                        prompt += "Q: " + convert_to_sent(disambig_list[i][0]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                        prompt += "Q: " + convert_to_sent(disambig_list[i][1]).strip() + "\r\n" + "A: FALSE" + "\r\n"
            else:  # Put Context False first
                for i in range(tmp_shots):
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                if (num_disambig > 0):
                    print("F/T NOT POSSIBLE")
                    for i in range(len(disambig_list)):
                        prompt += "Q: " + convert_to_sent(disambig_list[i][1]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                        prompt += "Q: " + convert_to_sent(disambig_list[i][0]).strip() + "\r\n" + "A: TRUE" + "\r\n"

            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
            prompt = prompt.strip()
            # expect either TRUE or FALSE as the answer
            # *** DOESNT WORK ANYMORE, DIFFERENT API CALL NOW. WORKED BEFORE. ***
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
                    (prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.'), prompt_lst[8 + i * 18]))
                prompt_dict["context"].append(
                    (prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.'), prompt_lst[17 + i * 18]))
            prompt_dict["query"].append((prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            data_file.write(str(prompt_dict))
            data_file.write("\n")
            data_file.write(whole)
            data_file.write("\n")
            completed_queries += 1
            # time.sleep(3.1)

    data_file.close()
# Return true if the sentence contains a human, false if it contains an animal

#  # multiple_context_requests(2, 2, False)
#  # multiple_context_requests(3, 2, False)
#  # multiple_context_requests(4, 2, False)
#  # multiple_context_requests(5, 2, False)


def random_multiple_context_requests(shots, order, ambig=True, num_disambig=0):
    encoding = 'utf-8'
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # ambig_set is comprised of human/nature and animal/urban pairings from ambig sets.
    pick_set = set()
    if ambig:
        pick_set.update(generate_x_y_requests(human_query, nature_query),
                        generate_x_y_requests(animal_query, urban_query))
    else:
        pick_set.update(generate_x_y_requests(human_query, urban_query),
                        generate_x_y_requests(animal_query, nature_query))
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("unbalanced_sanitycheck" + str(shots) + "shots_" +
                     "order_" + str(order) + "_" +
                     ("ambig_" if ambig else "NOTambig_") +
                     ("disambig_" + str(num_disambig) + "_" if num_disambig > 0 else "NONEdisambig_") +
                     "responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = 200
    completed_queries = 0
    # creates a 2(shots) + 1 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    usedLists = set()
    while completed_queries < num_queries:
        context = []
        ctxt_total = human_urban_ctxt | animal_nature_ctxt
        tmp_shots = shots - num_disambig
        for i in range(2 * tmp_shots):
            ctx = random.choice(tuple(ctxt_total))
            while ctx in context:
                ctx = random.choice(tuple(ctxt_total))
            context.append(ctx)
        pick = random.choice(
            tuple(pick_set))  # All ambiguous/non-ambiguous combos, we figure out correctness in data.py
        if order == 2:
            totalContext = context
            for i in range(num_disambig):
                # add human disambig first
                totalContext.append(
                    Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))))
                totalContext.append(
                    Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context))))
            # check TF alternating pattern
            random.shuffle(totalContext)
            altCheck = True
            if shots != 1:
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
                if totalContext[i][0] in human_context:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                else:
                    prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"

            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
            prompt = prompt.strip()

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
            prompt_lst = prompt.split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots):
                prompt_dict["context"].append(
                    (prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.'), prompt_lst[8 + i * 18]))
                prompt_dict["context"].append(
                    (prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.'), prompt_lst[17 + i * 18]))
            prompt_dict["query"].append((prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            cleaned_str = str(response["choices"]).replace("\n", "").replace("      ", " ").replace("    ", " ") \
                .replace("   ", " ").replace("  ", " ").replace("{ ", "{").replace(" }", "}").replace("[ ", "[") \
                .replace(" ]", "]").partition(": ")[2].removesuffix("]")
            data_file.write(cleaned_str)
            data_file.write("\n")
            completed_queries += 1
            # time.sleep(3.1)

        else:  # alternate = true
            if (num_disambig > 0):
                print("not possible")
                disambig_list = []
                while (tuple(contextTrue), tuple(contextFalse), tuple(disambig_list), pick) in usedLists:
                    for i in range(num_disambig):
                        disambig_list.append(
                            (Request(random.choice(tuple(human_context)), random.choice(tuple(nature_context))),
                             Request(random.choice(tuple(animal_context)), random.choice(tuple(urban_context)))))
                if (tuple(contextTrue), tuple(contextFalse), disambig_list, pick) in usedLists:
                    continue
                usedLists.add((tuple(contextTrue), tuple(contextFalse), disambig_list, pick))
            else:
                if (tuple(contextTrue), tuple(contextFalse), pick) in usedLists:
                    continue
                usedLists.add((tuple(contextTrue), tuple(contextFalse), pick))
            prompt = ""
            if order == 1:  # Put Context True first
                for i in range(tmp_shots):
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                if (num_disambig > 0):
                    print("T/F NOT POSSIBLE")
                    for i in range(len(disambig_list)):
                        prompt += "Q: " + convert_to_sent(disambig_list[i][0]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                        prompt += "Q: " + convert_to_sent(disambig_list[i][1]).strip() + "\r\n" + "A: FALSE" + "\r\n"
            else:  # Put Context False first
                for i in range(tmp_shots):
                    prompt += "Q: " + convert_to_sent(contextFalse[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                    prompt += "Q: " + convert_to_sent(contextTrue[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
                if (num_disambig > 0):
                    print("F/T NOT POSSIBLE")
                    for i in range(len(disambig_list)):
                        prompt += "Q: " + convert_to_sent(disambig_list[i][1]).strip() + "\r\n" + "A: FALSE" + "\r\n"
                        prompt += "Q: " + convert_to_sent(disambig_list[i][0]).strip() + "\r\n" + "A: TRUE" + "\r\n"

            prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
            prompt = prompt.strip()
            # expect either TRUE or FALSE as the answer
            # *** DOESNT WORK ANYMORE, DIFFERENT API CALL NOW. WORKED BEFORE. ***
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
                    (prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.'), prompt_lst[8 + i * 18]))
                prompt_dict["context"].append(
                    (prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.'), prompt_lst[17 + i * 18]))
            prompt_dict["query"].append((prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            data_file.write(str(prompt_dict))
            data_file.write("\n")
            data_file.write(whole)
            data_file.write("\n")
            completed_queries += 1
            # time.sleep(3.1)

    data_file.close()

#   #  random_multiple_context_requests(2, 2, False)
#   #  random_multiple_context_requests(3, 2, False)
#   #  random_multiple_context_requests(4, 2, False)
#   #  random_multiple_context_requests(5, 2, False)

def second_random_learning(shots):
    encoding = 'utf-8'
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    CONST_MISMATCH, CONST_MATCH = set(), set()
    # ambig
    CONST_MISMATCH.update(generate_x_y_requests(human_query, nature_query),
                          generate_x_y_requests(animal_query, urban_query))
    # non-ambig
    CONST_MATCH.update(generate_x_y_requests(human_query, urban_query),
                       generate_x_y_requests(animal_query, nature_query))
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("second_RandomLearning_OPENAI_" + str(shots) + "shots_" +
                     "responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = 400
    completed_queries = 0
    # creates a 2(shots) + 1 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    usedLists = set()
    while completed_queries < num_queries:
        # start with 1 context pair
        # 50/50 chance of True/False or False/True initial context
        if random.choice([0, 1]) == 0:
            context = [random.choice(list(human_urban_ctxt)), random.choice(list(animal_nature_ctxt))]
        else:
            context = [random.choice(list(animal_nature_ctxt)), random.choice(list(human_urban_ctxt))]
        rand_context = []
        for i in range(shots):
            if random.random() < 0.2:  # mismatch
                ctx = random.choice(tuple(CONST_MISMATCH))
                while ctx in rand_context:
                    ctx = random.choice(tuple(CONST_MISMATCH))
                rand_context.append(ctx)
            else:  # match
                ctx = random.choice(tuple(CONST_MATCH))
                while ctx in rand_context:
                    ctx = random.choice(tuple(CONST_MATCH))
                rand_context.append(ctx)
        random.shuffle(rand_context)
        totalContext = context + rand_context
        pick = totalContext.pop(-1)
        # check TF alternating pattern
        altCheck = True
        if shots != 1:
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
            if totalContext[i][0] in human_context or totalContext[i][0] in human_query:
                prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: TRUE" + "\r\n"
            else:
                prompt += "Q: " + convert_to_sent(totalContext[i]).strip() + "\r\n" + "A: FALSE" + "\r\n"

        prompt += "Q: " + convert_to_sent(pick).strip() + "\r\n" + "A: "
        prompt = prompt.strip()

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
        cleaned_str = str(response["choices"]).replace("\n", "").replace("      ", " ").replace("    ", " ") \
            .replace("   ", " ").replace("  ", " ").replace("{ ", "{").replace(" }", "}").replace("[ ", "[") \
            .replace(" ]", "]").partition(": ")[2].removesuffix("]")
        data_file.write(cleaned_str)
        data_file.write("\n")
        completed_queries += 1
        # time.sleep(3.1)
    data_file.close()

# second_random_learning(8)