import json


def to_str(i):
    if isinstance(i, int):
        if i < 10:
            return '00' + str(i)
        elif i < 100:
            return '0' + str(i)
        else:
            return str(i)
    else:
        raise ValueError('Is not an int')

maxi = 0
home = '/Users/adenhandasyde/GitHub/EEG/'
for c in range(64):
    for t in range(11):
        importing_file = home + 'Results/122/' + to_str(c) + '/' + to_str(t) + '.json'
        with open(importing_file, 'rb') as fp:
            inputs = json.load(fp)
        if inputs > maxi:
            maxi = inputs

print(maxi)
print(c)
print(t)

tally = 0
for c in range(64):
    t = 10
    importing_file = home + 'Results/122/' + to_str(c) + '/' + to_str(t) + '.json'
    with open(importing_file, 'rb') as fp:
        inputs = json.load(fp)
    tally += inputs

print(tally/64)
© 2022 GitHub, Inc.
Terms
Privacy
