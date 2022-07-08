from Bio.Seq import Seq
from itertools import product
import csv
import os
import argparse
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="directory with sample files",required=True)
    parser.add_argument("-s", "--sample", help="sample to analyze",required=True)
    parser.add_argument("-r", "--results",help="results dir",required=False)

    args = parser.parse_args()

    return args

args = get_args()

kmers = product(['A','T','C','G'], repeat = 3)
kmers = [kmer for kmer in kmers]
kmers = [''.join([i for i in kmer]) for kmer in kmers]

gc_lst = np.linspace(0.2,0.8,61)
gc_intervals = []
for index,item in enumerate(gc_lst):
    try:
        gc_intervals.append('%s-%s' % (round(gc_lst[index],3),round(gc_lst[index+1],3)))
    except:
        break

kmer_bins = {}
for idx in range(len(gc_intervals)):
	kmer_bins[idx] = {}
	kmer_bins[idx]['3'] = {kmer:0 for kmer in kmers}
	kmer_bins[idx]['5'] = {kmer:0 for kmer in kmers}

input_fh = os.path.join(args.dir,'%s.fragments' % args.sample)
with open(input_fh,'r') as mydata:
    for line in mydata:
        chr,start,end,length,gc,sequence = [item.strip() for item in line.split('\t')]

        idx = np.digitize(float(gc),gc_lst) - 1
        if idx == -1:
            continue
        try:
            gc_bin = gc_intervals[idx]
        except:
            print('gc of %f is out of bounds' % float(gc))
            continue

    	#-- forward (+, reference) strand fragments --#
        sequence = sequence.upper()
        if int(length) > 0:
            #-- 3' kmer --#
            kmer = sequence[-3:]
            if 'N' not in kmer:
                kmer_bins[idx]['3'][kmer] += 1

            #-- 5' kmer --#
            kmer = sequence[:3]
            if 'N' not in kmer:
                kmer_bins[idx]['5'][kmer] += 1

        #-- reverse (-) strand fragments --#
        elif int(length) < 0:
            #-- 5' kmer --3
            kmer = str(Seq(sequence[-3:]).reverse_complement())
            if 'N' not in kmer:
                kmer_bins[idx]['5'][kmer] += 1

            #-- 3' kmer --#
            kmer = str(Seq(sequence[:3]).reverse_complement())
            if 'N' not in kmer:
                kmer_bins[idx]['3'][kmer] += 1

output = open(os.path.join(args.results,'%s.mendseq.txt' % args.sample),'w+')
writer = csv.writer(output,delimiter='\t')
writer.writerow(['3mer','GC Bin','Count','Frequency'])
for idx,gc_bin in enumerate(kmer_bins):
    total = np.sum([kmer_bins[idx]['5'][kmer] for kmer in kmers])
    for kmer in kmers:
        #-- take the average of the 5' and 3' ends --#
        average = (kmer_bins[idx]['5'][kmer]+kmer_bins[idx]['3'][str(Seq(kmer).reverse_complement())])/2
        frequency = average/total
        data = [kmer,gc_intervals[idx],average,frequency]
        writer.writerow(data)
output.close()
