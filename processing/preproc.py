import numpy as np 
import pandas as pd 
import os
import json
import scipy.stats
from collections import Counter
import spacy
import wordcloud
import sys
import argparse
sys.path.append('langid.py/')
import langid

def unzip_files(path_to_zip, path_to_dir, quiet = False):
    if quiet:
        cm = ''
    else:
        cm = '-q'
    cmd = f'unzip {cm} {path_to_zip} -d {path_to_dir}'
    os.system(cmd)

def get_total_lengths(dir, total_num_docs = None):
    docs = os.listdir(dir)
    
    doc_lengths = []
    for i, doc in tqdm(enumerate(docs)):
        # get full directory of json file
        if total_num_docs is None:
            pass
        else:
            if i >= total_num_docs:
                return doc_lengths

        full_path = f'{dir}/{doc}'
        with open(full_path) as f:
            js = json.load(f)
            doc_len = 0
            for t in js['body_text']:
                doc_len += len(t['text'])
            doc_lengths.append(doc_len)

    doc_length_df = pd.DataFrame({'sha':docs, 'length': doc_lengths})

    return doc_length_df

def identify_language(metadata):
    df = metadata.copy()
    langs = []
    for title in tqdm(df.title.tolist()):       
        if isinstance(title ,str):
            lang, _ = langid.classify(title)
        else:
            lang = 'unknown'
        langs.append(lang)
    df['langs'] = langs

    return df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('zip_path', type = str, help ='path to zipped cord-19 dataset.')
    parser.add_argument('path_to_dir', type = str,help = 'path to store unzipped files')
    parser.add_argument('doc_src_type', choices = ['pdf', 'pmc'])
    parser.add_argument('-n','--doc_length', type = int, help = 'maximum document length')
    parser.add_argument('-q', '--quiet', type=bool, nargs='?', default=True, help= 'do not print out processing.')

    args = parser.parse_args()

    # load language model from spacy
    nlp = spacy.load('en_core_web_sm')

    print(args.doc_src_type)

    unzip_files(path_to_zip = args.zip_path, path_to_dir = args.path_to_dir, 
    quiet=args.quiet)

    # get path to metadata
    metadata = pd.read_csv(f'{path_to_dir}/metadata.csv')

    # get document lengths
    




