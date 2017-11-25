import sys
import fileinput
import re
import matplotlib.pyplot as plt

REGEX = "\(\(([-\d.]+),([-\d.]+)\),([-\d.]+)\)"

def main():
    Xs = []
    Ys = []
    colors = []
    for line in fileinput.input():
        match = re.match(REGEX, line)
        if not match or len(match.groups()) != 3:
            continue
        x = match.groups()[0]
        y = match.groups()[1]
        c = match.groups()[2]
        Xs.append(x)
        Ys.append(y)
        colors.append(c)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
    plt.scatter(Xs, Ys, c=colors, alpha=0.5)
    ax.add_patch(circ)
    plt.show()



if __name__ == "__main__":
    main()
