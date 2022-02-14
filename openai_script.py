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

human_ambig = {"human", "toddler", "woman", "doctor", "firefighter", "soldier", "banker", "actor", "architect",
               "butcher", "engineer", "student"}
animal_ambig = {"hawk", "elephant", "ant", "mouse", "crocodile", "shark", "sheep", "lion", "salamander", "bee",
                "condor", "chipmunk"}  # "buffalo", "panda"}
urban_ambig = {"skyscraper", "restaurant", "alley", "store", "apartment", "condominium", "house", "office", "museum",
               "casino", "hospital", "library"}  # "airport"}
nature_ambig = {"ocean", "tundra", "forest", "cave", "canyon", "lake", "stream", "savannah", "stream", "creek", "delta",
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


# Return true if the sentence contains a human, false if it contains an animal
def multiple_context_requests(shots, order, ambig=True, num_disambig=0):
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
                     ("disambig_" + str(num_disambig) + "_" if num_disambig > 0 else "NONEdisambig_") +
                     "responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    num_queries = NUM_QUERIES
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


def active_learning(shots, order, ambig=True, num_disambig=0):
    """
    In this case, shots refers to how many active learning cycles we do until we stop, starting at 1 context pair
    of human/animal correct examples. I don't think order matters. I don't think ambig matters. I don't think
    num_disambig matters.
    """
    human_urban_ctxt = generate_x_y_requests(human_context, urban_context)
    animal_nature_ctxt = generate_x_y_requests(animal_context, nature_context)
    # start with 1 context pair
    context = [random.choice(list(human_urban_ctxt)), random.choice(list(animal_nature_ctxt))]
    ambig_pick_set, disambig_pick_set = set(), set()
    # ambig is the mismatches
    ambig_pick_set.update(generate_x_y_requests(human_ambig, nature_ambig), generate_x_y_requests(animal_ambig, urban_ambig))
    # disambig is the correct matches
    disambig_pick_set.update(generate_x_y_requests(human_ambig, urban_ambig), generate_x_y_requests(animal_ambig, nature_ambig))
    actual_time = strftime("%Y-%m-%dT_%H-%M-%SZ", gmtime())
    data_file = open("ActiveLearning_OPENAI_" + str(shots) + "shots_" +
                     "order_" + str(order) + "_" +
                     ("ambig_" if ambig else "NOTambig_") +
                     ("disambig_" + str(num_disambig) + "_" if num_disambig > 0 else "NONEdisambig_") +
                     "responses_" + actual_time + ".txt", 'w', encoding="utf-8")
    # begin iterations
    curr_shot = 1
    while curr_shot <= shots:
        # store probabilities to find smallest abs difference later
        examples_probs = {}
        # format context and examples to presentable tokens
        # create 4 ambig examples
        examples = []
        for i in range(4):
            examples.append(ambig_pick_set.pop())
        # create 1 dis-ambig examples
        examples.append(disambig_pick_set.pop())
        context_prompt = ""
        # make context prompt randomly
        random_context = random.sample(context, len(context))
        for ctxt in random_context:
            context_prompt += "Q: " + convert_to_sent(ctxt).strip() + "\r\n" + "A: " + \
                              ("TRUE" if (ctxt.thing in human_context or ctxt.thing in human_ambig) else "FALSE") + "\r\n"
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
            raw_F, raw_T = math.exp(cleaned_dct["logprobs"]["top_logprobs"][-1][" FALSE"]), math.exp(cleaned_dct["logprobs"]["top_logprobs"][-1][" TRUE"])
            # normalize them
            norm_F, norm_T = raw_F / (raw_F + raw_T), raw_T / (raw_F + raw_T)
            # absolute difference of normalized probabilities
            examples_probs[example] = abs(norm_T - norm_F)
        # find true smallest example (smallest difference in probabilities)
        smallest_example = examples[0]
        smallest_prob = examples_probs[smallest_example]
        for example in examples_probs:
            if smallest_prob > examples_probs[example]:
                smallest_example = example
                smallest_prob = examples_probs[smallest_example]
        # add to context, "Active Learning"
        context.append(smallest_example)
            # .504 .486
            # (.504 - .486)/(.504 + .486)
            # normalized and then absolute difference (Margin sampling)
            # .504 - .486
            # we want absolute difference that is smallest (most ambiguous response from model)
            # shuffle into context, repeat
            # for each of those 5 examples, we are storing the results for each
            # accuracy of the performance of the model
            # shuffle for each new example
            # percent difference, not prob between first two responses
            # FIX SKELETON BELOW
            # does the model pick the dis-ambig example more than random chance?
            # does the accuracy on both ambig and disambig examples stay high?
            # does the performance on disambig examples increase?
            # as you acquire more examples, do you do better on ambiguous examples using active learning or random picking
            # random only needs 1 call, at the end
        curr_shot += 1


# active_learning(1, 1)
# multiple_context_request(#shots, random, ambig=True, #disambig)
# multiple_context_requests(2, 2, True, 0)  # 20 DONE
# multiple_context_requests(2, 2, True, 1)  # 31 DONE

# multiple_context_requests(3, 2, True, 0)  # 21 DONE
# multiple_context_requests(3, 2, True, 1)  # 32 DONE
# multiple_context_requests(3, 2, True, 2)  # 36 DONE

# multiple_context_requests(4, 2, True, 0)  # 22 DONE
# multiple_context_requests(4, 2, True, 1)  # 33 DONE
# multiple_context_requests(4, 2, True, 2)  # 37 DONE
# multiple_context_requests(4, 2, True, 3)  # 40 DONE

# multiple_context_requests(5, 2, True, 0)  # 23 DONE
# multiple_context_requests(5, 2, True, 1)  # 34 DONE
# multiple_context_requests(5, 2, True, 2)  # 38 DONE
# multiple_context_requests(5, 2, True, 3)  # 41 DONE
# multiple_context_requests(5, 2, True, 4)  # 43 DONE

# multiple_context_requests(1, 0, True, 0)  # 6 DONE
# multiple_context_requests(2, 0, True, 0)  # 7 DONE
# multiple_context_requests(3, 0, True, 0)  # 8 DONE

# multiple_context_requests(1, 1, True, 0)  # 1 DONE
# multiple_context_requests(2, 1, True, 0)  # 2 DONE
# multiple_context_requests(3, 1, True, 0)  # 3 DONE

# multiple_context_requests(1, 0, False, 0)  # 14 DONE
# multiple_context_requests(2, 0, False, 0)  # 15 DONE
# multiple_context_requests(3, 0, False, 0)  # 16 DONE

# multiple_context_requests(1, 1, False, 0)  # 10 DONE
# multiple_context_requests(2, 1, False, 0)  # 11 DONE
# multiple_context_requests(3, 1, False, 0)  # 12 DONE

# multiple_context_requests(1, 2, True, 0)  # 19 DONE

# multiple_context_requests(1, 2, False, 0)  # 25 DONE
# multiple_context_requests(2, 2, False, 0)  # 26 DONE
# multiple_context_requests(3, 2, False, 0)  # 27 DONE
# multiple_context_requests(4, 2, False, 0)  # 28 DONE
# multiple_context_requests(5, 2, False, 0)  # 29 DONE

# multiple_context_requests(2, 2, False, 1)  # 45 DONE
# multiple_context_requests(3, 2, False, 1)  # 46 DONE
# multiple_context_requests(4, 2, False, 1)  # 47 DONE
# multiple_context_requests(5, 2, False, 1)  # 48 DONE

# multiple_context_requests(3, 2, False, 2)  # 50 DONE
# multiple_context_requests(4, 2, False, 2)  # 51 DONE
# multiple_context_requests(5, 2, False, 2)  # 52 DONE

# multiple_context_requests(4, 2, False, 3)  # 54 DONE
# multiple_context_requests(5, 2, False, 3)  # 55 DONE

# multiple_context_requests(5, 2, False, 4)  # 57 DONE

# multiple_context_requests(4, 0, True, 0) # DONE
# multiple_context_requests(5, 0, True, 0) # DONE

# multiple_context_requests(4, 1, True, 0) # DONE
# multiple_context_requests(5, 1, True, 0) # DONE

# multiple_context_requests(4, 0, False, 0) # DONE
# multiple_context_requests(5, 0, False, 0) # DONE

# multiple_context_requests(4, 1, False, 0) # DONE
# multiple_context_requests(5, 1, False, 0) # DONE
