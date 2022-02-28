import ast
import openai_script
import math
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def new_pls():
    filename = "RANDOMLearning_OPENAI_8shots_responses_20_runs_2022-02-27T_06-16-57Z.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        df = {}
        data = f.readlines()
        two_cycle = 0
        run = 0
        for line in data:
            if two_cycle == 0:
                two_cycle += 1
                ret = json.loads(line)
            else:
                run += 1
                two_cycle -= 1
                context_info = ast.literal_eval(line.strip())
                tokens = ret['logprobs']['tokens']
                top_logprobs = ret['logprobs']['top_logprobs']
                TRUE_indices = [i for i, token in enumerate(tokens) if token == " TRUE"]
                FALSE_indices = [i for i, token in enumerate(tokens) if token == " FALSE"]
                BOTH_indices = TRUE_indices + FALSE_indices
                BOTH_indices.sort()
                for i, idx in enumerate(BOTH_indices):
                    # skip the first TRUE Q/A and first FALSE Q/A
                    if i <= 1:
                        continue
                    key = str(run) + "_" + str(idx)
                    # check that " FALSE" and " TRUE" are the two keys
                    if " FALSE" not in top_logprobs[idx].keys() or " TRUE" not in top_logprobs[idx].keys():
                        print("FALSE/TRUE not in top 2 logprobs")
                        break
                    df[key] = [tokens[idx]]
                    df[key].append(math.exp(top_logprobs[idx][tokens[idx]]) / (math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"])))
                    # model predicted right
                    if df[key][1] > 0.5:
                        df[key].append(True)
                    else: # model predicted wrong
                        df[key].append(False)
                    # how many Q/A before this Q/A
                    df[key].append(i)
                    # if this Q/A was match or mismatch
                    df[key].append(context_info["match_type_info"][i])
        df = pd.DataFrame.from_dict(df, orient='index', columns=["token", "normedProbCorrect", "ifCorrect", "numResponsesBefore", "ifMatch"])
        df.to_csv('prelim_random_OPENAI.csv')

# new_pls()
def show_values(axs, orient="v", space=.01):
    """
    credits to Zach at statology.org for show_values
    https://www.statology.org/seaborn-barplot-show-values/
    """

    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + .015  # (p.get_height()*0.1)
                value = '{:.4f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", color="black")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                value = '{:.4f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def prelim():
    df = pd.read_csv('prelim_random_OPENAI.csv')
    print(df)
    print(df.groupby("numResponsesBefore")["ifMatch"].mean())
    ax = sns.barplot(x='numResponsesBefore', y='ifMatch', data=df, ci=None, palette="Blues_d")
    ax.set(xlabel="# of responses before", ylabel=" % of matched questions")
    show_values(ax)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))
    sns.color_palette("rocket")
    plt.show()

prelim()

def pls():
    # put whatever u want
    filename = open("responses_to_add/CURIE_1shots_order_1_NOTambig_NONEdisambig_responses_2021-12-05T_04-02-01Z.txt",
                    'r', encoding="utf-8")
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
    datadict = {"shots": [], "correct": [], "certainty": [], "subject": [], "order": [], "ambig": [], "disambig": [],
                "disambig_ratio": [], "model": []}
    # filename = open("OPENAI_1shots_order_0_ambig_NONEdisambig_responses_2021-11-26T_23-55-46Z.txt", 'r', encoding="utf-8")
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
                    datadict["disambig"].append(
                        0 if "NONE" in f.name else (int(f.name[f.name.find("disambig_") + len("disambig_")])))
                    datadict["ambig"].append(False if "NOT" in f.name else True)
                    datadict["order"].append(
                        "Random" if "order_2" in f.name else ("F/T" if "order_0" in f.name else "T/F"))
                else:
                    ans = ast.literal_eval(line.strip())
                    if human_flag:
                        if ans["choices"][0]["text"].strip() == "TRUE":
                            per_correct += 1
                        try:
                            per_certain_of_correct = math.exp(ans["choices"][0]["logprobs"]["top_logprobs"][0][" TRUE"])
                        except:
                            per_certain_of_correct = 0

                        if (per_certain_of_correct >= .5):
                            datadict["correct"].append(True)
                        else:
                            datadict["correct"].append(False)
                        datadict["certainty"].append(per_certain_of_correct)

                    if animal_flag:
                        if ans["choices"][0]["text"].strip() == "FALSE":
                            per_correct += 1
                        try:
                            per_certain_of_correct = math.exp(
                                ans["choices"][0]["logprobs"]["top_logprobs"][0][" FALSE"])
                        except:
                            per_certain_of_correct = 0

                        if (per_certain_of_correct >= .5):
                            datadict["correct"].append(True)
                        else:
                            datadict["correct"].append(False)
                        datadict["certainty"].append(per_certain_of_correct)
                    datadict["model"].append("Curie")
                    datadict["disambig_ratio"].append(num_disambig // (len(context) // 2))

    outf = open("SUPERDATAwCurie1", "w")
    tmpdf = pd.DataFrame(data=datadict)
    superdf = maindf.append(tmpdf, ignore_index=True)
    superdf.to_csv(outf)

# openai_df()
# pls()
