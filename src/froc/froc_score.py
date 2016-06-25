import csv
import sys
from collections import defaultdict

def froc_analyse(csvfile):
    averages = [0,0,0]
    FPrate = defaultdict(list)
    with open(csvfile, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for rows in reader:
            FPrate[float(rows[0])].extend(map(float, rows[1:4]))
    for a in range(-3, 4):
        closest = min(FPrate.keys(), key=lambda x:abs(x-2**a))
        averages = [x + y for x,y in zip(averages, FPrate.get(closest))]
    averages[:] = [x / 7 for x in averages]
    print averages

if __name__ == "__main__":
    froc_analyse(sys.argv[1])
