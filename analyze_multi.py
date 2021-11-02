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
#import base_script as bs
import os


def analyze():
    filename = open("2shots_random_ambig_disambig_responses_2021-11-02T_05-56-30Z.txt", encoding="utf-8")
    data = filename.readlines()
    for line in data:
        line = ast.literal_eval(line)
        # print(line["completions"])
        wtf = []
        print(len(line["completions"][0]["data"]["tokens"][0]["topTokens"]))
        for elem in line["completions"][0]["data"]["tokens"][0]["topTokens"]:
            wtf.append((elem["token"], round(math.exp(elem["logprob"]), 3)))
        total_prob = 0
        for elem in wtf:
            total_prob += elem[1]
        print(wtf)
        print(total_prob)


analyze()
