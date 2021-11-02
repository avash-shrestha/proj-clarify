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


def analyze():
    for filename in os.listdir("multishot responses\\2shots\\"):
        with open("multishot responses\\2shots\\" + filename, "r", encoding="utf8") as f:
            data = f.readlines()
            numTrue, numFalse = 0, 0
            numCorrect, numIncorrect = 0, 0
            numHuman, numAnimal = 0, 0
            avg_percent_certain = 0
            for line in data:
                line = ast.literal_eval(line)
                firstResponse = (line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["token"].strip('▁'),
                                 math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][0]["logprob"]))
                secondResponse = (line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["token"].strip('▁'),
                                 math.exp(line["completions"][0]["data"]["tokens"][0]["topTokens"][1]["logprob"]))
                # print(firstResponse, secondResponse)
                normalizingProb = firstResponse[1] + secondResponse[1]
                prompt = line['prompt']['text'].split()
                # number of shots
                numShots = int(f.name[20])
                isDisambig = False if "NONEdisambig" in f.name else True
                index = 18 * numShots + 3 + (18 if isDisambig else 0) - 1
                if prompt[index] in bs.human_ambig:
                    numHuman += 1
                    if firstResponse[0] == "TRUE":
                        numTrue += 1
                        numCorrect += 1
                        percent_certain = firstResponse[1]
                    else:
                        numFalse += 1
                        numIncorrect += 1
                        percent_certain = secondResponse[1]
                else:  # in animal_ambig
                    numAnimal += 1
                    if firstResponse[0] == "TRUE":
                        numTrue += 1
                        numIncorrect += 1
                        percent_certain = secondResponse[1]
                    else:
                        numFalse += 1
                        numCorrect += 1
                        percent_certain = firstResponse[1]
                percent_certain = percent_certain / normalizingProb
                avg_percent_certain += percent_certain
            print(f)
            print(avg_percent_certain / 300)
            print(numHuman, numAnimal, numHuman + numAnimal)
            print(numTrue, numFalse, numTrue + numFalse)
            print(numCorrect, numIncorrect, numCorrect + numIncorrect)


analyze()
