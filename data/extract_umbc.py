"""
This script turns the UMBC News webcorpus into a file where
each line is a sentence the corpus.
In the resulting file, tokenization has already been performed and can be recreated by splitting by the whitespace character.

The script assumes as parameters:
arg[1] : Path to folder with .txt files
        The extracted UMBC webcorpus consists of a bunch of .txt files with a paragraph in each line (with blank
        lines in between.

arg[2] : Output file path
        Path to the location where the output file should be stored (i.e., the file containing one sentence per line).
"""

import sys, os
from nltk.tokenize import sent_tokenize, word_tokenize


# parse args
input_path = sys.argv[1]
output_file = sys.argv[2]

with open(output_file, 'w') as output_f:


    file_names = os.listdir(input_path)
    file_names = [f for f in file_names if f.endswith(".txt")]

    for f in file_names:
        input_file_path = os.path.join(input_path, f)

        # iterate over the plain text data from the UMBC corpus
        with open(input_file_path, 'r', encoding = "utf-8") as input_file:
            for line in input_file:

                # skip empty line
                if not line.strip():
                    continue
                else:
                    line = line.strip()

                # get all sentences in paragraph
                sentences = sent_tokenize(line)

                # each line contains a sentence
                for s in sentences:
                    words = word_tokenize(s)
                    sample = " ".join(words)
                    print(sample, file = output_f)
