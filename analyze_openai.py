import ast
import openai_script
import math
import os
import pandas as pd
import json

def new_pls():
    filename = "ActiveLearning_OPENAI_1shots_order_1_ambig_NONEdisambig_responses_2022-02-14T_01-31-04Z.txt"
    with open(filename, "r", encoding="utf-8") as f:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            raw_F, raw_T = math.exp(line["logprobs"]["top_logprobs"][-1][" FALSE"]), math.exp(
                line["logprobs"]["top_logprobs"][-1][" TRUE"])
            norm_F, norm_T = raw_F / (raw_F + raw_T), raw_T / (raw_F + raw_T)
            print(line["text"])
            print(raw_F, raw_T)
            print(norm_F, norm_T)
    """data = json.load(filename)
    print(type(data))
    print(data["logprobs"].keys())
    print(data["logprobs"]["top_logprobs"][-1])
    raw_F, raw_T = math.exp(data["logprobs"]["top_logprobs"][-1][" FALSE"]), math.exp(data["logprobs"]["top_logprobs"][-1][" TRUE"])
    norm_F, norm_T = raw_F / (raw_F + raw_T), raw_T / (raw_F + raw_T)
    print(raw_F, raw_T)
    print(norm_F, norm_T)
    print(len(data["logprobs"]["top_logprobs"]))"""
    # data = filename.readlines()
    # print(type(data[5]))
    #query = ast.literal_eval(data[5].strip())
new_pls()
def pls():
    # put whatever u want
    filename = open("responses_to_add/CURIE_1shots_order_1_NOTambig_NONEdisambig_responses_2021-12-05T_04-02-01Z.txt", 'r', encoding="utf-8")
    data = filename.readlines()
    per_correct = 0.0
    per_certain_of_correct = 0.0
    human_flag = False
    animal_flag = False
    wtf = 0
    for i, line in enumerate(data):
        if i % 2 == 0:
            print(line)
            query = ast.literal_eval(line.strip())["query"]
            print(query)
            if query[0][0] in openai_script.animal_ambig:
                human_flag = False
                animal_flag = True
            if query[0][0] in openai_script.human_ambig:
                human_flag = True
                animal_flag = False
        else:
            print(line)
            break
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
# pls()
def openai_df(): 
    inf = open("SUPERDATA_COMBINE2", "r")
    maindf = pd.read_csv(inf, index_col=0)
    datadict = {"shots": [], "correct": [], "certainty": [], "subject": [], "order": [], "ambig" : [], "disambig" : [], "disambig_ratio" : [], "model" : []}
    #filename = open("OPENAI_1shots_order_0_ambig_NONEdisambig_responses_2021-11-26T_23-55-46Z.txt", 'r', encoding="utf-8")
    for filename in os.listdir("responses_to_add"):
        with open("responses_to_add\\" + filename, "r", encoding="utf8") as f:
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
                        try: 
                            per_certain_of_correct = math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" TRUE"])
                        except: 
                            per_certain_of_correct = 0

                        if(per_certain_of_correct >= .5):
                            datadict["correct"].append(True)
                        else: 
                            datadict["correct"].append(False)
                        datadict["certainty"].append(per_certain_of_correct)

                    if animal_flag:
                        if ans["choices"][0]["text"].strip() == "FALSE":
                            per_correct += 1
                        try:
                            per_certain_of_correct = math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" FALSE"])
                        except: 
                            per_certain_of_correct = 0


                        if(per_certain_of_correct >= .5):
                            datadict["correct"].append(True)
                        else: 
                            datadict["correct"].append(False)
                        datadict["certainty"].append(per_certain_of_correct)
                    datadict["model"].append("Curie")
                    datadict["disambig_ratio"].append(num_disambig // (len(context) // 2))
           
    outf = open("SUPERDATAwCurie1", "w")
    tmpdf = pd.DataFrame(data = datadict)
    superdf = maindf.append(tmpdf, ignore_index= True)
    superdf.to_csv(outf)


# openai_df()
#pls()
