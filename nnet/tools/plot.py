from bokeh.plotting import figure, show, output_file
import fileinput
import numpy as np
import re
import sys

REGEX = "\(\(([-\d.]+),([-\d.]+)\),([-\d.]+)\)"

def main():
    Xs = []
    Ys = []
    colors = []
    for line in fileinput.input():
        match = re.match(REGEX, line)
        if not match or len(match.groups()) != 3:
            continue
        x = float(match.groups()[0])
        y = float(match.groups()[1])
        score = match.groups()[2]
        score_intensity = int(255 * float(score))
        color = "#%02x%02x%02x" % (score_intensity, score_intensity,
        255)
        Xs.append(x)
        Ys.append(y)
        colors.append(color)
    radii = np.array([0.1 for i in Xs])
    x = np.array(Xs)
    y = np.array(Ys)

    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

    p = figure(tools=TOOLS, match_aspect=True)
    p.scatter(x, y, marker="circle", radius = radii, fill_color = colors,
    fill_alpha = 1, line_color = None)
    output_file("/tmp/graph.html", title="Circle Test Visualization")
    show(p)


main()
