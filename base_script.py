# This script assumes that a string using a word from human_list is TRUE and a string using a word from animal_list
# is false. The ambiguous pairs (human/nature and animal/urban) are tested for correctness on AI21 Jurassic-Jumbo.
from collections import namedtuple
import json
import requests
import time
import random

# Trevor's api
# api_key = "KEArioXJKgpnEkhDLQbLBiGdfe0a8Knq"
# Avash's api
api_key = "nGAafrSAtOgan0kUVLepbhelY6HMeMJr"


# thing is either human_list or animal_list
# place is either urban_list or nature_list
# result is either 1 (TRUE), 2 (FALSE), 3 (TRUE), or 4 (FALSE)
# # # 1 and 2 are context
# # # 3 and 4 are ambiguous
Request = namedtuple("Request", ["thing", "place", "result"])

# FUTURE: make list of namedtuples
template = ["The ", " is in a "]

human_list = {"person", "human", "man", "woman", "child"}
animal_list = {"tiger", "iguana", "toad", "butterfly", "rhinoceros"}
urban_list = {"skyscraper", "building", "city", "dumpster", "street"}
nature_list = {"meadow", "forest", "pond", "tree", "river"}


def generate_human_urban_requests():
    """
     Generates and returns a set of human/urban Requests with a result of 1 (Context TRUE)
    """
    request_set = set()
    for human in human_list:
        for urban in urban_list:
            request_set.add(Request(human, urban, 1))
    return request_set


def generate_animal_nature_requests():
    """
     Generates and returns a set of animal/nature Requests with a result of 2 (Context FALSE)
    """
    request_set = set()
    for animal in animal_list:
        for nature in nature_list:
            request_set.add(Request(animal, nature, 2))
    return request_set


def generate_human_nature_requests():
    """
     Generates and returns a set of human/nature Requests with a result of 3 (Ambig TRUE)
    """
    request_set = set()
    for human in human_list:
        for nature in nature_list:
            request_set.add(Request(human, nature, 3))
    return request_set


def generate_animal_urban_requests():
    """
     Generates and returns a set of animal/urban Requests with a result of 4 (Ambig FALSE)
    """
    request_set = set()
    for animal in animal_list:
        for urban in urban_list:
            request_set.add(Request(animal, urban, 4))
    return request_set


def convert_to_sent(r):
    """
    Makes and returns an English sentence from the pair of words passed in, r.
    """
    return template[0] + r.thing + template[1] + r.place + "."


# Return true if the sentence contains a human, false if it contains an animal
def query_requests_1():
    # human_urban and animal_nature are the context sets
    human_urban = generate_human_urban_requests()
    animal_nature = generate_animal_nature_requests()
    # ambig_set is comprised of human/nature and animal/urban pairings.
    ambig_set = set()
    ambig_set.update(generate_human_nature_requests(), generate_animal_urban_requests())
    data_file = open("response1.txt", 'a')
    # creates a 3 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    for hu in human_urban:
        for an in animal_nature:
            ambig = random.choice(tuple(ambig_set))
            prompt = "Q: " + convert_to_sent(hu) + "\n A: TRUE \n Q: " + convert_to_sent(
                an) + "\n A: FALSE \n Q: " + convert_to_sent(ambig) + "\n A:"
            # expect either TRUE or FALSE as the answer
            response = str(requests.post(
                "https://api.ai21.com/studio/v1/j1-jumbo/complete",
                headers={"Authorization": "Bearer " + api_key},
                json={
                    "prompt": prompt,
                    "numResults": 1,
                    "maxTokens": 1,
                    "stopSequences": ["."],
                    "topKReturn": 0,
                    "temperature": 0.0
                }
            ).json())
            data_file.write(response)
            data_file.write("\n")
            if response == "{'detail': 'Quota exceeded.'}":
                break
            time.sleep(3.5)


query_requests_1()
