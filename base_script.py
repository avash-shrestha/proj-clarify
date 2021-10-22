# This script assumes that a string using a word from human_list is TRUE and a string using a word from animal_list
# is false. The ambiguous pairs (human/nature and animal/urban) are tested for correctness on AI21 Jurassic-Jumbo.
from collections import namedtuple
import json
import requests
import time, random
import ast
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import date, datetime
#1. percent of the time the guess is correct
#2. How often it is correct for each ambigous type (compare result = 3 to result = 4)

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

human_list = {"person", "child", "man", "officer"}
animal_list = {"tiger", "iguana", "toad", "butterfly"}
urban_list = {"theater", "building", "city", "street"}
nature_list = {"meadow", "mountain", "pond", "desert"}

human_ambig = {"human", "toddler", "woman", "doctor"}
animal_ambig = {"hawk", "elephant", "ant", "mouse"}
urban_ambig = {"skyscraper", "restaurant", "alley", "store"}
nature_ambig = {"ocean", "hills", "forest", "cave"}


def generate_human_urban_requests_context():
    """
     Generates and returns a set of human/urban Requests with a result of 1 (Context TRUE)
    """
    request_set = set()
    for human in human_list:
        for urban in urban_list:
            request_set.add(Request(human, urban, 1))
    return request_set


def generate_animal_nature_requests_context():
    """
     Generates and returns a set of animal/nature Requests with a result of 2 (Context FALSE)
    """
    request_set = set()
    for animal in animal_list:
        for nature in nature_list:
            request_set.add(Request(animal, nature, 2))
    return request_set


def generate_human_nature_requests_ambig():
    """
     Generates and returns a set of human/nature Requests with a result of 3 (Ambig TRUE)
    """
    request_set = set()
    for human in human_ambig:
        for nature in nature_ambig:
            request_set.add(Request(human, nature, 3))
    return request_set


def generate_animal_urban_requests_ambig():
    """
     Generates and returns a set of animal/urban Requests with a result of 4 (Ambig FALSE)
    """
    request_set = set()
    for animal in animal_ambig:
        for urban in urban_ambig:
            request_set.add(Request(animal, urban, 4))
    return request_set

def convert_to_sent(r):
    """
    Makes and returns an English sentence from the pair of words passed in, r.
    """
    return template[0] + r.thing + template[1] + r.place + "."


# Return true if the sentence contains a human, false if it contains an animal
def query_ambig_requests_1():
    # human_urban and animal_nature are the context sets
    human_urban = generate_human_urban_requests_context()
    animal_nature = generate_animal_nature_requests_context()
    # ambig_set is comprised of human/nature and animal/urban pairings.
    ambig_set = set()
    ambig_set.update(generate_human_nature_requests_ambig(), generate_animal_urban_requests_ambig())
    data_file = open("responses_ambig_" + date.today().strftime("%m-%d-%y") + datetime.now().strftime("%d/%m/%Y %H%M%S"), 'a', encoding = "utf-8" )
    used_set = set()
    num_queries = 1000; 
    completed_queries = 0

    # creates a 3 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    while(completed_queries < num_queries):
        c1 = random.choice(tuple(human_urban))
        c2 = random.choice(tuple(animal_nature))
        ambig = random.choice(tuple(ambig_set))
        while((c1, c2, ambig) in used_set):
            c1 = random.choice(tuple(human_urban))
            c2 = random.choice(tuple(animal_nature))
            ambig = random.choice(tuple(ambig_set))
        used_set.add((c1,c2,ambig))                  
        prompt = "Q: " + convert_to_sent(c1) + "\n A: TRUE \n Q: " + convert_to_sent(
            c2) + "\n A: FALSE \n Q: " + convert_to_sent(ambig) + "\n A: "
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
        data_file.write(response)
        data_file.write("\n")
        if response == "{'detail': 'Quota exceeded.'}":
            break
        completed_queries += 1
        time.sleep(3.1)

def query_non_ambig_requests(): 
    human_urban = generate_human_urban_requests_context()
    animal_nature = generate_animal_nature_requests_context()
    # ambig_set is comprised of human/nature and animal/urban pairings.
    ambig_set = set()
    ambig_set.update(human_urban, animal_nature)
    data_file = open("responses_nambig_" + datetime.now().strftime("%d-%m-%Y %H-%M-%S"), 'a', encoding = "utf-8" )
    used_set = set()
    num_queries = 1000; 
    completed_queries = 0

    # creates a 3 Q/A prompt, with the first 2 being from each of the context sets and the last being just a Q
    # from the ambig_set
    while(completed_queries < num_queries):
        c1 = random.choice(tuple(human_urban))
        c2 = random.choice(tuple(animal_nature))
        ambig = random.choice(tuple(ambig_set))
        while((c1, c2, ambig) in used_set):
            c1 = random.choice(tuple(human_urban))
            c2 = random.choice(tuple(animal_nature))
            ambig = random.choice(tuple(ambig_set))
        used_set.add((c1,c2,ambig))                  
        prompt = "Q: " + convert_to_sent(c1) + "\n A: TRUE \n Q: " + convert_to_sent(
            c2) + "\n A: FALSE \n Q: " + convert_to_sent(ambig) + "\n A: "
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
        data_file.write(response)
        data_file.write("\n")
        if response == "{'detail': 'Quota exceeded.'}":
            break
        completed_queries += 1
        time.sleep(3.1)




# NEW **********************************************************************
def analyze_correct_vs_incorrect(): 
    c_vs_inc = {"Correct" : [], "Incorrect" : []}
    with open("response_fixed.txt", "r", encoding = "utf-8") as f:
        data = f.readlines()
        for i in range(len(data)): 
            temp = ast.literal_eval(data[i])
            prompt = temp["prompt"]["text"]
            ambigWord = prompt.split()[20]
            completion = temp["completions"][0]["data"]["text"].strip()
            prob = math.exp(temp["completions"][0]["data"]["tokens"][0]["generatedToken"]["logprob"])
            result = (prompt, completion, prob)
            if ambigWord in human_ambig:
                if completion == "TRUE":
                    c_vs_inc["Correct"].append(result)
                else:
                    c_vs_inc["Incorrect"].append(result)
            else:  # ambigWord is an animal
                if completion == "FALSE":
                    c_vs_inc["Correct"].append(result)
                else:
                    c_vs_inc["Incorrect"].append(result)

def analyze_ambig(): 
    #resultsFile = open("results_scrub.txt", "wb")
    with open("response_fixed.txt", "r", encoding = "utf-8") as f:
        results = {"human_true" : [], "human_false" : [], "animal_true": [], "animal_false": []}
        data = f.readlines()
        for i in range(len(data)): 
            temp = ast.literal_eval(data[i])
            prompt = temp["prompt"]["text"]
            ambigWord = prompt.split()[20]
            completion = temp["completions"][0]["data"]["text"].strip()
            prob = math.exp(temp["completions"][0]["data"]["tokens"][0]["generatedToken"]["logprob"])
            result = (prompt, completion, prob)
            if ambigWord in human_ambig:
                if completion == "TRUE":
                    results["human_true"].append(result)
                else:
                    results["human_false"].append(result)
            else:  # ambigWord is an animal
                if completion == "FALSE":
                    results["animal_false"].append(result)
                else:
                    results["animal_true"].append(result)

    #pickle.dump(results, resultsFile)

    #print((len(results["human_true"]) + len(results["animal_false"])) / (len(results["human_true"]) + len(results["animal_false"]) + len(results["human_false"]) + len(results["animal_true"]) ))
    #.4576
    #print(len(results["human_true"]) / (len(results["human_true"]) + len(results["human_false"])))
    #.8145
    #print(len(results["animal_false"]) / (len(results["animal_true"]) + len(results["animal_false"])))
    #.3693
    print(len(results["human_true"]) + len(results["human_false"]))
    print(len(results["animal_true"]) + len(results["animal_false"]))
    return results
    
def graph():
    # Importing the matplotlib library
    # Categorical data: Country names
    results = analyze_ambig()
    total_prob = (len(results["human_true"]) + len(results["animal_false"])) / (len(results["human_true"]) + len(results["animal_false"]) + len(results["human_false"]) + len(results["animal_true"]))
    human_prob = len(results["human_true"]) / (len(results["human_true"]) + len(results["human_false"]))
    animal_prob = len(results["animal_false"]) / (len(results["animal_true"]) + len(results["animal_false"]))
    # labels = ['total', 'human', 'animal']
    # prob_of_correct = [total_prob, human_prob, animal_prob]
    temp = [['total', total_prob], ['human', human_prob], ['animal', animal_prob]]
    # Integer value interms of death counts
    # Passing the parameters to the bar function, this is the main function which creates the bar plot
    """plt.bar(labels, prob_of_correct)
    # Displaying the bar plot
    plt.show()
    plt.close()"""
    df = pd.DataFrame(temp, columns=["Group", 'Probability of Success'])
    print(df)
    plt.figure(figsize=(14, 8))
    # plot a bar chart
    ax = sns.barplot(x="Group", y="Probability of Success", data=df, estimator=np.mean, ci=95, capsize=.2, color='lightblue')
    plt.show()

            
        



#query_requests_1()
#analyze_ambig()
graph()