import os
home = '/Users/adenhandasyde/GitHub/EEG/Transformed Data/'


def to_str(k):
    if k < 10:
        return '00' + str(k)
    elif k < 100:
        return '0' + str(k)
    else:
        return str(k)

files = [[0]*120]*160
for n in range(160):
    for j in range(120):

            # save list of actual files, rename so they're in order
            if os.path.exists(home + to_str(n) + '/'):
                if os.path.exists(home + to_str(n) + '/' + to_str(j) + '/'):
                    files[n][j] = 1
                else:
                    b = j + 1
                    while (not os.path.exists(home + to_str(n) + '/' + to_str(b) + '/')) and b < 118:
                        b += 1
                    if os.path.exists(home + to_str(n) + '/' + to_str(b) + '/'):
                        os.rename(home + to_str(n) + '/' + to_str(b) + '/',
                                  home + to_str(n) + '/' + to_str(j) + '/')
            else:
                o = n + 1
                while (not os.path.exists(home + to_str(o) + '/')) and o < 159:
                    o += 1
                if os.path.exists(home + to_str(o) + '/'):
                    os.rename(home + to_str(o) + '/',
                              home + to_str(n) + '/')


