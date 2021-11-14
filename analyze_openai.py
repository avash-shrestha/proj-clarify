import ast


def pls():
    # put whatever u want
    filename = open(".txt", 'r', encoding="utf-8")
    data = filename.readlines()
    i = 0
    for line in data:
        i += 1
        if i % 2 == 1:
            print(line)
        else:
            line = ast.literal_eval(line)
            print(line)
            print(type(line))
            print(line.keys())
            for key in line:
                print(key, ":", line[key])


pls()