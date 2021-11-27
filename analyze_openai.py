import ast
import openai_script
import math

def pls():
    # put whatever u want
    filename = open("OPENAI_1shots_order_0_ambig_NONEdisambig_responses_2021-11-26T_23-55-46Z.txt", 'r', encoding="utf-8")
    data = filename.readlines()
    per_correct = 0.0
    per_certain_of_correct = 0.0
    human_flag = False
    animal_flag = False
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


pls()
