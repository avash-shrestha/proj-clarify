import ast
import openai_script
import openai_winter2022
import math
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def nearestNeighbors():
    filename = 'second_RandomLearning_OPENAI_8shots_responses_2022-03-28T_08-02-27Z.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        shots = 10
        # make 94 entries, one for each entity/place and True/False
        # mark the average probability of each when an entry has been seen previously
        df = {"entity": [[0, 0], [0, 0]], "place": [[0, 0], [0, 0]]}
        data = f.readlines()
        run = 0
        for line in data:
            # reset nearestNeighbors every new trial
            nN = {"entity": {}, "place": {}}
            ret = json.loads(line)
            prompt_lst = ret['text'].split()
            prompt_dict = {"questions": []}
            for i in range(shots - 1):
                prompt_dict["questions"].append(
                    (prompt_lst[2 + i * 9], prompt_lst[6 + i * 9].strip('.'), prompt_lst[8 + i * 9]))
            prompt_dict["questions"].append((prompt_lst[2 + (shots - 1) * 9], prompt_lst[6 + (shots - 1) * 9].strip('.'), prompt_lst[8 + (shots - 1) * 9]))
            # print(prompt_dict)
            run += 1
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
                # check that " FALSE" and " TRUE" are the two keys
                if " FALSE" not in top_logprobs[idx].keys() or " TRUE" not in top_logprobs[idx].keys():
                    print("FALSE/TRUE not in top 2 logprobs")
                    return
                entity, place, answer = prompt_dict["questions"][i]
                # [bool last, [cumul prob of true last, cumul prob of false last], [# of true last, # of false last]
                if entity not in nN["entity"]:
                    nN["entity"][entity] = [answer, [0, 0], [0, 0]]
                else:
                    if nN["entity"][entity][0] == "TRUE":
                        nN["entity"][entity][1][0] += math.exp(top_logprobs[idx][' TRUE']) / (
                                    math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))
                        nN["entity"][entity][2][0] += 1
                    elif nN["entity"][entity][0] == "FALSE":
                        nN["entity"][entity][1][1] += math.exp(top_logprobs[idx][' FALSE']) / (
                                math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))
                        nN["entity"][entity][2][1] += 1
                    else:
                        print('not True/False for entity')
                        return
                    # change last seen to answer
                    nN["entity"][entity][0] = answer
                if place not in nN["place"]:
                    nN["place"][place] = [answer, [0, 0], [0, 0]]
                else:
                    if nN["place"][place][0] == "TRUE":
                        nN["place"][place][1][0] += math.exp(top_logprobs[idx][' TRUE']) / (
                                    math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))
                        nN["place"][place][2][0] += 1
                    elif nN["place"][place][0] == "FALSE":
                        nN["place"][place][1][1] += math.exp(top_logprobs[idx][' FALSE']) / (
                                math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))
                        nN["place"][place][2][1] += 1
                    else:
                        print('not True/False for place')
                        return
                    # change last seen to answer
                    nN["place"][place][0] = answer
            # put new nN entity and place into overall df
            for entity in nN["entity"].keys():
                df["entity"][0][0] += nN["entity"][entity][1][0]
                df["entity"][0][1] += nN["entity"][entity][1][1]
                df["entity"][1][0] += nN["entity"][entity][2][0]
                df["entity"][1][1] += nN["entity"][entity][2][1]
            for place in nN["place"].keys():
                df["place"][0][0] += nN["place"][place][1][0]
                df["place"][0][1] += nN["place"][place][1][1]
                df["place"][1][0] += nN["place"][place][2][0]
                df["place"][1][1] += nN["place"][place][2][1]
        # normalize df
        for key in df:
            df[key][0][0] = df[key][0][0] / df[key][1][0]
            df[key][0][1] = df[key][0][1] / df[key][1][1]
        print(df)

def organize():
    filename = "RANDOMLearning_OPENAI_8shots_responses_150_runs_2022-03-03T_03-40-07Z.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        datafile = open("justresponses_RANDOMLearning_OPENAI_8shots_responses_150_runs_2022-03-03T_03-40-07Z.txt", 'w', encoding='utf-8')
        data = f.readlines()
        two_cycle = 0
        for line in data:
            if two_cycle == 0:
                two_cycle += 1
                datafile.write(line)
            else:
                two_cycle -= 1
# organize()
def tmp():
    filename = "second_RandomLearning_OPENAI_8shots_responses_2022-03-28T_08-02-27Z.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        """shots = 5
        data = f.readlines()
        num_correct = 0
        avg_correct_prob = 0
        for line in data:
            line = json.loads(line)
            if " FALSE" not in line['logprobs']['top_logprobs'][-1].keys():
                print("ERROR: FALSE NOT IN TOP 2 LOGPROBS")
                return
            if " TRUE" not in line['logprobs']['top_logprobs'][-1].keys():
                print("ERROR: TRUE NOT IN TOP 2 LOGPROBS")
                return

            prompt_lst = line['text'].split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots):
                prompt_dict["context"].append(
                    (prompt_lst[2 + i * 18], prompt_lst[6 + i * 18].strip('.'), prompt_lst[8 + i * 18]))
                prompt_dict["context"].append(
                    (prompt_lst[11 + i * 18], prompt_lst[15 + i * 18].strip('.'), prompt_lst[17 + i * 18]))
            prompt_dict["query"].append((prompt_lst[2 + shots * 18], prompt_lst[6 + shots * 18].strip('.')))
            answer_text = line['logprobs']['tokens'][-1]
            false_prob = math.exp(line['logprobs']['top_logprobs'][-1][' FALSE'])
            true_prob = math.exp(line['logprobs']['top_logprobs'][-1][' TRUE'])
            if prompt_dict['query'][0][0] in openai_winter2022.human_query:
                corr_ans_prob = math.exp(line['logprobs']['top_logprobs'][-1][' TRUE']) / (false_prob + true_prob)
                if answer_text == ' TRUE':
                    num_correct += 1
                avg_correct_prob += corr_ans_prob
            elif prompt_dict['query'][0][0] in openai_winter2022.animal_query:
                corr_ans_prob = math.exp(line['logprobs']['top_logprobs'][-1][' FALSE']) / (false_prob + true_prob)
                if answer_text == ' FALSE':
                    num_correct += 1
                avg_correct_prob += corr_ans_prob
            else:
                print("ERROR: Not in human_query or animal_query")"""
        """shots = 10
        data = f.readlines()
        num_shot_dict = {}
        for line in data:
            line = json.loads(line)
            if " FALSE" not in line['logprobs']['top_logprobs'][-1].keys():
                print("ERROR: FALSE NOT IN TOP 2 LOGPROBS")
                return
            if " TRUE" not in line['logprobs']['top_logprobs'][-1].keys():
                print("ERROR: TRUE NOT IN TOP 2 LOGPROBS")
                return
            prompt_lst = line['text'].split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots - 1):
                prompt_dict["context"].append(
                    (prompt_lst[2 + i * 9], prompt_lst[6 + i * 9].strip('.'), prompt_lst[8 + i * 9]))
            prompt_dict["query"].append((prompt_lst[2 + (shots - 1) * 9], prompt_lst[6 + (shots - 1) * 9].strip('.')))
            tokens = line['logprobs']['tokens']
            top_logprobs = line['logprobs']['top_logprobs']
            TRUE_indices = [i for i, token in enumerate(tokens) if token == " TRUE"]
            FALSE_indices = [i for i, token in enumerate(tokens) if token == " FALSE"]
            BOTH_indices = TRUE_indices + FALSE_indices
            BOTH_indices.sort()
            for i, idx in enumerate(BOTH_indices):
                if i <= 1:
                    continue
                if " FALSE" not in top_logprobs[idx].keys() or " TRUE" not in top_logprobs[idx].keys():
                    print("FALSE/TRUE not in top 2 logprobs")
                    return
                if i not in num_shot_dict:
                    num_shot_dict[i] = [0, 0]
                example_prob_correct = math.exp(top_logprobs[idx][tokens[idx]]) / (math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))
                if i == len(BOTH_indices) - 1:
                    if prompt_dict["query"][0][0] in openai_winter2022.human_query:
                        example_prob_correct = math.exp(top_logprobs[idx][' TRUE']) / ((math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"])))
                    elif prompt_dict['query'][0][0] in openai_winter2022.animal_query:
                        example_prob_correct = math.exp(top_logprobs[idx][' FALSE']) / ((math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"])))
                    else:
                        print('NOT IN HUMAN QUERY OR ANIMAL QUERY')
                        return
                num_shot_dict[i][0] += example_prob_correct
                if example_prob_correct > 0.5:
                    num_shot_dict[i][1] += 1
    for key in num_shot_dict:
        num_shot_dict[key][0] = num_shot_dict[key][0] / 200
        num_shot_dict[key][1] = num_shot_dict[key][1] / 200
    print(num_shot_dict)"""

    # print(f"{shots} shots: num_correct = {num_correct / 200}, avg_correct_prob = {avg_correct_prob / 200}")
# tmp()
def new_new_pls():
    filename = "second_RandomLearning_OPENAI_8shots_combined.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        shots = 10
        df = {}
        data = f.readlines()
        run = 0
        for line in data:
            ret = json.loads(line)
            prompt_lst = ret['text'].split()
            prompt_dict = {"context": [], "query": []}
            for i in range(shots - 1):
                prompt_dict["context"].append(
                    (prompt_lst[2 + i * 9], prompt_lst[6 + i * 9].strip('.'), prompt_lst[8 + i * 9]))
            prompt_dict["query"].append((prompt_lst[2 + (shots - 1) * 9], prompt_lst[6 + (shots - 1) * 9].strip('.')))
            run += 1
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
                    return
                df[key] = [tokens[idx]]
                if i != len(BOTH_indices) - 1:
                    df[key].append(math.exp(top_logprobs[idx][tokens[idx]]) / (
                            math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"])))
                else:  # i is the last one aka the pick, so the text TRUE/FALSE isn't guaranteed to be right
                    if prompt_dict["query"][0][0] in openai_winter2022.human_query:
                        df[key].append(math.exp(top_logprobs[idx][' TRUE']) / ((math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))))
                    elif prompt_dict['query'][0][0] in openai_winter2022.animal_query:
                        df[key].append(math.exp(top_logprobs[idx][' FALSE']) / ((math.exp(top_logprobs[idx][" TRUE"]) + math.exp(top_logprobs[idx][" FALSE"]))))
                    else:
                        print('NOT IN HUMAN QUERY OR ANIMAL QUERY')
                        return
                # model predicted right
                if df[key][1] > 0.5:
                    df[key].append(True)
                else:  # model predicted wrong
                    df[key].append(False)
                # how many Q/A before this Q/A
                df[key].append(i)
                # if this Q/A was match or mismatch
                if i != len(BOTH_indices) - 1:
                    if prompt_dict["context"][i][0] in openai_winter2022.human_query:
                        if prompt_dict["context"][i][1] in openai_winter2022.urban_query:
                            df[key].append(True)
                        elif prompt_dict["context"][i][1] in openai_winter2022.nature_query:
                            df[key].append(False)
                        else:
                            print("not match or mismatch")
                            return
                    elif prompt_dict["context"][i][0] in openai_winter2022.animal_query:
                        if prompt_dict["context"][i][1] in openai_winter2022.urban_query:
                            df[key].append(False)
                        elif prompt_dict["context"][i][1] in openai_winter2022.nature_query:
                            df[key].append(True)
                        else:
                            print("not match or mismatch")
                            return
                    else:
                        print("not human_query or animal_query")
                        return
                else:
                    if prompt_dict["query"][0][0] in openai_winter2022.human_query:
                        if prompt_dict["query"][0][1] in openai_winter2022.urban_query:
                            df[key].append(True)
                        elif prompt_dict["query"][0][1] in openai_winter2022.nature_query:
                            df[key].append(False)
                        else:
                            print("not match or mismatch")
                            return
                    elif prompt_dict["query"][0][0] in openai_winter2022.animal_query:
                        if prompt_dict["query"][0][1] in openai_winter2022.urban_query:
                            df[key].append(False)
                        elif prompt_dict["query"][0][1] in openai_winter2022.nature_query:
                            df[key].append(True)
                        else:
                            print("not match or mismatch")
                            return
                    else:
                        print("not human_query or animal_query")
                        return
                # if the first two context Q/A are in human/urban -> animal/nature order (True -> False)
                if prompt_dict["context"][0][0] in openai_winter2022.human_context:
                    df[key].append(True)
                else:
                    # if the first two context Q/A are in animal/nature -> human/urban order (False -> True)
                    df[key].append(False)
        df = pd.DataFrame.from_dict(df, orient='index',
                                    columns=["token", "normedProbCorrect", "ifCorrect", "numResponsesBefore", "ifMatch",
                                             "ifContextHumanFirst"])
        df.to_csv('second_Random_OPENAI.csv')
# new_new_pls()
def new_pls():
    filename = "RANDOMLearning_OPENAI_8shots_responses_150_runs_2022-03-03T_03-40-07Z.txt"
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
                        return
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
                    # if the first two context Q/A are in human/urban -> animal/nature order (True -> False)
                    if context_info["context_info"][0][0] in openai_winter2022.human_context:
                        df[key].append(True)
                    else:
                        # if the first two context Q/A are in animal/nature -> human/urban order (False -> True)
                        df[key].append(False)
        df = pd.DataFrame.from_dict(df, orient='index', columns=["token", "normedProbCorrect", "ifCorrect", "numResponsesBefore", "ifMatch", "ifContextHumanFirst"])
        df.to_csv('sanity_random_OPENAI.csv')

# new_pls()
def active_pls():
    filename = "ActiveLearning_OPENAI_8shots_responses_20_runs_2022-03-03T_04-12-04Z.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        df = {}
        data = f.readlines()
        six_cycle = 1
        run = 0
        ret_lst = []
        for line in data:
            if six_cycle != 6:
                ret_lst.append(json.loads(line))
                six_cycle += 1
            else:
                run += 1
                six_cycle = 1
                context_info = ast.literal_eval(line.strip())
                query_info = context_info["query_info"]
                for query in query_info.keys():
                    # columns=["token", "normedProbCorrect", "ifCorrect", "numResponsesBefore", "ifMatch", "ifContextHumanFirst", ifPicked]
                    key = str(run) + "_" + query[0] + "_" + query[1]
                    df[key] = []
                    if query[0] in openai_winter2022.human_query:
                        df[key].append(" TRUE")
                        df[key].append(query_info[query][1] / (query_info[query][0] + query_info[query][1]))
                    else:
                        # animal prompt
                        df[key].append(" FALSE")
                        df[key].append(query_info[query][0] / (query_info[query][1] + query_info[query][0]))
                    if df[key][1] > 0.5:
                        df[key].append(True)
                    else:
                        df[key].append(False)
                    df[key].append(len(context_info["context_info"]))
                    df[key].append(query_info[query][2])
                    if context_info["context_info"][0][0] in openai_winter2022.human_context:
                        df[key].append(True)
                    else:
                        df[key].append(False)
                    if query == context_info["picked_example"][0]:
                        df[key].append(True)
                    else:
                        df[key].append(False)
        df = pd.DataFrame.from_dict(df, orient='index', columns=["token", "normedProbCorrect", "ifCorrect", "numResponsesBefore", "ifMatch", "ifContextHumanFirst", "ifPicked"])
        df.to_csv('tmp_active_OPENAI.csv')
# active_pls()
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
    df = pd.read_csv('second_Random_OPENAI.csv')
    print(df['normedProbCorrect'])
    df["Confidence"] = df.apply(lambda row: abs(0.5 - row.normedProbCorrect), axis = 1)
    ax = sns.barplot(x='numResponsesBefore', y='ifCorrect', hue='ifMatch', data=df, palette="Blues_d", ci=None)
    ax.set(xlabel="# of responses before", ylabel=" Percentage of Correct Answer")
    show_values(ax)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))
    plt.title("Random Learning Results (600 Responses)")
    sns.color_palette("rocket")
    plt.show()

# prelim()

def pls():
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
