import json
import re
import argparse

# parse argumants from command line
parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='folder with the data')
parser.add_argument('--data', help='filename of the data file')
parser.add_argument('--corpus', help='filename of the corpus file')

args = parser.parse_args()
folder = args.folder
filename_data = folder+'/'+args.data
filename_corpus = folder+'/'+args.corpus

# create a corpus with the words from the data and save it into filename_out
def create_corpus(filename_in, filename_out):
    with open(filename_in,'r') as f_in:
        with open(filename_out,'w') as f_out:
            for line in f_in:
                f_out.write(json.loads(line)['reviewText']+'\n')
                f_out.write(json.loads(line)['summary']+'\n')

# create corpus with words from the training data
create_corpus(filename_data, filename_corpus)
