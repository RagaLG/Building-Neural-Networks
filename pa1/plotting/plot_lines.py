#libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import csv

if len(sys.argv) < 3:
    print "Error, usage: python <infile_1 .. infile_n> <outfile>";
    exit(1)

infiles = sys.argv[1:-1];
outfile = sys.argv[-1];

print "infiles:", infiles
print "outfile:", outfile

# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

fileinfo = []

filenumber = 0
for infile in infiles:
    x = []
    y1 = []
    y2 = []

    print "infile: ", infile
    with open(infile) as fp:
        line = fp.readline()
        line = line[:-1] #get rid of the trailing newline

        fileinfo.append(line)
        line = fp.readline()
        line = line[:-1] #get rid of the trailing newline

        while line:
            print line
            row = line.split(' ')
            x.append(float(row[0]))
            y1.append(float(row[1]))
            y2.append(float(row[2]))

            line = fp.readline()
            line = line[:-1] #get rid of the trailing newline

        #print "x: ", x
        #print "y1: ", y1
        #print "y2: ", y2

        plt.plot(x, y1, color=palette((filenumber * 2)), linewidth=0.1, alpha=0.9, label="best " + fileinfo[filenumber])
        plt.plot(x, y2, color=palette((filenumber * 2) + 1), linewidth=0.1, alpha=0.9, label="curr " + fileinfo[filenumber])

    filenumber += 1

# Add legend
plt.legend(loc=1, ncol=1)

# Add titles
plt.title("", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(outfile, dpi=300)
