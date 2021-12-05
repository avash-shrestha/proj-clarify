import ast
import openai_script
import math
import os
import pandas as pd

def pls():
    # put whatever u want
    filename = open("CURIE_1shots_order_1_NOTambig_NONEdisambig_responses_2021-12-05T_04-02-01Z.txt", 'r', encoding="utf-8")
    data = filename.readlines()
    per_correct = 0.0
    per_certain_of_correct = 0.0
    human_flag = False
    animal_flag = False
    wtf = 0
    for i, line in enumerate(data):
        if i % 2 == 0:
            query = ast.literal_eval(line.strip())["query"]
            if query[0][0] in openai_script.animal_ambig:
                human_flag = False
                animal_flag = True
            if query[0][0] in openai_script.human_ambig:
                human_flag = True
                animal_flag = False
        else:
            ans = ast.literal_eval(line.strip())
            if ans["choices"][0]["text"].strip() == "FALSE":
                wtf += 1
            if human_flag:
                if ans["choices"][0]["text"].strip() == "TRUE":
                    per_correct += 1
                per_certain_of_correct += math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" TRUE"])
            if animal_flag:
                if ans["choices"][0]["text"].strip() == "FALSE":
                    per_correct += 1
                per_certain_of_correct += math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" FALSE"])
    print(per_correct / 600)
    print(per_certain_of_correct / 600)
    print(wtf)
"""
def openai_df(): 
    datadict = {"shots": [], "correct": [], "certainty": [], "subject": [], "order": [], "ambig" : [], "disambig" : []}
    #filename = open("OPENAI_1shots_order_0_ambig_NONEdisambig_responses_2021-11-26T_23-55-46Z.txt", 'r', encoding="utf-8")
    for filename in os.listdir("openai_responses"):
        with open("openai_responses\\" + filename, "r", encoding="utf8") as f:
            data = f.readlines()
            per_correct = 0.0
            per_certain_of_correct = 0.0
            human_flag = False
            animal_flag = False
            num_disambig = 0
            for i, line in enumerate(data):
                num_disambig = 0
                if i % 2 == 0:
                    query = ast.literal_eval(line.strip())["query"]
                    context = ast.literal_eval(line.strip())["context"]
                    if query[0][0] in openai_script.animal_ambig:
                        human_flag = False
                        animal_flag = True
                        datadict["subject"].append("animal")
                    if query[0][0] in openai_script.human_ambig:
                        human_flag = True
                        animal_flag = False
                        datadict["subject"].append("human")
                    
                    datadict["shots"].append(len(context) // 2)
                    datadict["disambig"].append(0 if "NONE" in f.name else (int(f.name[f.name.find("disambig_") + len("disambig_")])))
                    datadict["ambig"].append(False if "NOT" in f.name else True)
                    datadict["order"].append("Random" if "order_2" in f.name else ("F/T" if "order_0" in f.name else "T/F"))
                else:
                    ans = ast.literal_eval(line.strip())
                    if human_flag:
                        if ans["choices"][0]["text"].strip() == "TRUE":
                            per_correct += 1
                        per_certain_of_correct = math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" TRUE"])
                        if(per_certain_of_correct >= .5):
                            datadict["correct"].append(True)
                        else: 
                            datadict["correct"].append(False)
                        datadict["certainty"].append(per_certain_of_correct)

                    if animal_flag:
                        if ans["choices"][0]["text"].strip() == "FALSE":
                            per_correct += 1
                        per_certain_of_correct = math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" FALSE"])
                        if(per_certain_of_correct >= .5):
                            datadict["correct"].append(True)
                        else: 
                            datadict["correct"].append(False)
                        datadict["certainty"].append(per_certain_of_correct)

    df = pd.DataFrame(data = datadict)
    outf = open("openai_df", "w")
    df.to_csv(outf)
"""

# openai_df()
pls()
