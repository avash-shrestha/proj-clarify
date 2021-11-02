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
from pandas.io.parsers import read_csv
import requests
import seaborn as sns
import base_script as bs
import os

# FILENAME = "responses_2021-10-25T_03-09-31Z.txt"
'''fin = open("test_dataframe__responses_2021-10-25T_07-40-27Z.csv", "rb")
df = pd.read_csv(fin)
g = sns.FacetGrid(df, col= "is_ambig")
g.map(sns.barplot, "is_human", "is_correct", alpha=.7)
g.add_legend()
plt.show()'''


def split_dataframes():
    numCorrect_dict = {"value": [], 'label': []}
    certaintyCorrect_dict = {"value": [], 'label': []}
    # numCorrect_dict = {"T_T_T" : [], "T_T_F" : [], "T_F_T":[], "T_F_F":[], "F_T_T":[], "F_T_F":[], "F_F_T":[], "F_F_F":[]}
    # certaintyCorrect_dict =  {"T_T_T" : [], "T_T_F" : [], "T_F_T":[], "T_F_F":[], "F_T_T":[], "F_T_F":[], "F_F_T":[], "F_F_F":[]}

    for filename in os.listdir("dataframe_files\\main_dfs"):
        # filename = "test_dataframe__responses_2021-10-25T_03-09-31Z.csv"
        with open("dataframe_files\\main_dfs\\" + filename, "r") as f:
            df = read_csv(f)
            df_ambig = df[df["is_ambig"] == True]
            df_nambig = df[df["is_ambig"] == False]

            ambig_certainty = df_ambig.groupby(["is_context_true_first", "is_human"]).mean().drop(
                columns=["Unnamed: 0", "is_ambig"])

            nambig_certainty = df_nambig.groupby(["is_context_true_first", "is_human"]).mean().drop(
                columns=["Unnamed: 0", "is_ambig"])

            '''numCorrect_dict["T_F_F"].append(ambig_certainty.iat[0, 0])
            numCorrect_dict["T_F_T"].append(ambig_certainty.iat[1, 0])
            numCorrect_dict["T_T_F"].append(ambig_certainty.iat[2, 0])
            numCorrect_dict["T_T_T"].append(ambig_certainty.iat[3, 0])'''
            numCorrect_dict["value"].append(ambig_certainty.iat[0, 0])
            numCorrect_dict["label"].append("T_F_F")
            numCorrect_dict["value"].append(ambig_certainty.iat[1, 0])
            numCorrect_dict["label"].append("T_F_T")
            numCorrect_dict["value"].append(ambig_certainty.iat[2, 0])
            numCorrect_dict["label"].append("T_T_F")
            numCorrect_dict["value"].append(ambig_certainty.iat[3, 0])
            numCorrect_dict["label"].append("T_T_T")

            certaintyCorrect_dict["value"].append(ambig_certainty.iat[0, 1])
            certaintyCorrect_dict["label"].append("T_F_F")

            certaintyCorrect_dict["value"].append(ambig_certainty.iat[1, 1])
            certaintyCorrect_dict["label"].append("T_F_T")

            certaintyCorrect_dict["value"].append(ambig_certainty.iat[2, 1])
            certaintyCorrect_dict["label"].append("T_T_F")

            certaintyCorrect_dict["value"].append(ambig_certainty.iat[3, 1])
            certaintyCorrect_dict["label"].append("T_T_T")

            numCorrect_dict["value"].append(nambig_certainty.iat[0, 0])
            numCorrect_dict["label"].append("F_F_F")
            numCorrect_dict["value"].append(nambig_certainty.iat[1, 0])
            numCorrect_dict["label"].append("F_F_T")
            numCorrect_dict["value"].append(nambig_certainty.iat[2, 0])
            numCorrect_dict["label"].append("F_T_F")
            numCorrect_dict["value"].append(nambig_certainty.iat[3, 0])
            numCorrect_dict["label"].append("F_T_T")

            certaintyCorrect_dict["value"].append(nambig_certainty.iat[0, 1])
            certaintyCorrect_dict["label"].append("F_F_F")

            certaintyCorrect_dict["value"].append(nambig_certainty.iat[1, 1])
            certaintyCorrect_dict["label"].append("F_F_T")

            certaintyCorrect_dict["value"].append(nambig_certainty.iat[2, 1])
            certaintyCorrect_dict["label"].append("F_T_F")

            certaintyCorrect_dict["value"].append(nambig_certainty.iat[3, 1])
            certaintyCorrect_dict["label"].append("F_T_T")

    numCorrect_frame = pd.DataFrame(data=numCorrect_dict)
    percentCertain_frame = pd.DataFrame(data=certaintyCorrect_dict)
    numCorrectFile = open("numCorrectMainDF", "w")
    numCorrect_frame.to_csv(numCorrectFile)
    certaintyCorrectFile = open("certaintyCorrectMainDF", "w")
    percentCertain_frame.to_csv(certaintyCorrectFile)


# split_dataframes()
def show_values(axs, orient="v", space=.01):
    """
    credits to Zach at statology.org for show_values
    https://www.statology.org/seaborn-barplot-show-values/
    """
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + .015 #(p.get_height()*0.1)
                value = '{:.4f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", color="black")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.4f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

def scatter():
    certaintyDF = pd.read_csv(open("certaintyCorrectMainDF", "r"))
    numCorrectDF = pd.read_csv(open("numCorrectMainDF", "r"))
    # df1 = certaintyDF.query("label == 'F_T_F' or label == 'F_T_T'")

    g = sns.barplot(data=certaintyDF, x="label", y="value")
    plt.text(0.75, 0.75,
             'x_y_z\nx: T if example is ambiguous,\n    F if example is not ambiguous\ny: T if context ordering is T->F\n    F if context ordering is F->T\nz: T if human example\n    F if animal example',
             color='black',
             bbox=dict(facecolor='darkgrey', edgecolor='black', boxstyle='round,pad=1'))
    # plt.text(.75,.75,"x_y_z\nx: T if example is ambiguous,\n    F if example is not ambiguous\ny: T if context ordering is T->F\n    F if context ordering is F->T\nz: T if human example\n    F if animal example")
    plt.title("Percent Certainty of Correct Response by Query Category")
    plt.xlabel("Category")
    plt.ylabel("% Certainty")
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))
    show_values(g)
    plt.show()

def tmp():
    data = sns.load_dataset("tips")
    p = sns.barplot(x="day", y="tip", data=data, ci=None).set_ybound

    # show values on barplot
    print(type(p))
    return
    show_values(p)
# tmp()

scatter()


# temp()
def analyze_generic():
    for filename in os.listdir("responses"):
        with open("responses\\" + filename, "r", encoding="utf8") as f:
            data = f.readlines()
            tmp_set = set()
            datadict = {"is_ambig": [], "is_correct": [], "is_human": [], "is_context_true_first": [],
                        "percent_certainty_of_correct": []}
            for line in data:
                is_correct = False
                is_ambig = False
                is_human = False
                is_context_true_first = False
                line = ast.literal_eval(line)
                completion = line["completions"][0]["data"]["text"].strip()
                prompt = line["prompt"]["text"].split()
                # 2 is first thing, 6 is first place, 8 is first answer
                # 11 is second thing, 15 is second place, 17 is second answer
                # 20 is third thing, 24 is third place
                # first check if ambig or not by comparing third thing and thrid place
                # if ambig20 is in human and ambig24 is nature or ambig20 is in animal and ambig24 is in urban
                if (prompt[20] in bs.human_ambig and prompt[24].strip('.') in bs.nature_ambig) or (
                        prompt[20] in bs.animal_ambig and prompt[24].strip('.') in bs.urban_ambig):
                    is_ambig = True
                if (prompt[20] in bs.human_ambig and completion == "TRUE") or (
                        prompt[20] in bs.animal_ambig and completion == "FALSE"):
                    is_correct = True
                if prompt[20] in bs.human_ambig:
                    is_human = True
                if prompt[8] == "TRUE":
                    is_context_true_first = True

                first_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["token"].strip()
                second_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["token"].strip()
                tmp_set.add(first_token)
                tmp_set.add(second_token)
                first_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["logprob"])
                second_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["logprob"])
                normalizing_prob = first_prob + second_prob  # since there is small percentage of answer that isn't T/F
                percent_certain = first_prob / normalizing_prob if is_correct else second_prob / normalizing_prob

                datadict["is_ambig"].append(is_ambig)
                datadict["is_correct"].append(is_correct)
                datadict["is_human"].append(is_human)
                datadict["is_context_true_first"].append(is_context_true_first)
                datadict["percent_certainty_of_correct"].append(percent_certain)

            df = pd.DataFrame(datadict)
            outf = open("dataframe_files\\test_dataframe__" + filename.strip(".txt") + ".csv", "wb")
            df.to_csv(outf)
            outf.close()
        f.close()
        print(tmp_set)


def analyze():
    with open(FILENAME, "r", encoding="utf8") as f:
        data = f.readlines()
        human_corr, human_not_corr, human_tot, avg_human_corr_prob = 0.0, 0.0, 0.0, 0.0
        animal_corr, animal_not_corr, animal_tot, avg_animal_corr_prob = 0.0, 0.0, 0.0, 0.0
        avg_prob = 0.0
        tmp_set = set()
        for line in data:
            if line[0] == '(':
                continue
            line = ast.literal_eval(line)
            completion = line["completions"][0]["data"]["text"].strip()
            prompt = line["prompt"]["text"].split()
            first_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["token"].strip()
            second_token = line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["token"].strip()
            tmp_set.add(first_token)
            tmp_set.add(second_token)
            first_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["logprob"])
            second_prob = math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["logprob"])
            normalizing_prob = first_prob + second_prob
            avg_prob += normalizing_prob
            if prompt[20] in base_human_ambig:  # want completion to be TRUE
                human_tot += 1
                if completion == "TRUE":
                    avg_human_corr_prob += first_prob / normalizing_prob
                    human_corr += 1
                else:
                    avg_human_corr_prob += second_prob / normalizing_prob
                    human_not_corr += 1
            else:  # prompt[20] in animal_ambig  # want completion to be FALSE
                animal_tot += 1
                if completion == "FALSE":
                    avg_animal_corr_prob += first_prob / normalizing_prob
                    animal_corr += 1
                else:
                    avg_animal_corr_prob += second_prob / normalizing_prob
                    animal_not_corr += 1
        print("HUMAN: ", human_corr, human_not_corr, human_tot, human_corr / human_tot)
        print("ANIMAL: ", animal_corr, animal_not_corr, animal_tot, animal_corr / animal_tot)
        print("PROBS: ", avg_human_corr_prob / human_tot, avg_animal_corr_prob / animal_tot,
              avg_prob / (human_tot + animal_tot))
        print(tmp_set)


# analyze_generic()
