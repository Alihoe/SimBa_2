import os
import shutil

import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move

from src.utils import delete_first_line_of_tsv, prepare_corpus_tsv


def run():
    file_url = requests.get("http://nlp.stanford.edu/data/glove.840B.300d.zip")
    with open("glove.840B.300d.zip", 'wb') as f:
        f.write(file_url.content)
    with ZipFile("glove.840B.300d.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, os.getcwd())
    os.remove("glove.840B.300d.zip")
    file_url = requests.get("https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
    with open("crawl-300d-2M.vec.zip", 'wb') as f:
        f.write(file_url.content)
    with ZipFile("crawl-300d-2M.vec.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, os.getcwd())
    os.remove("crawl-300d-2M.vec.zip")
    file_url = requests.get("https://dl.fbaipublicfiles.com/infersent/infersent1.pkl")
    with open("infersent/infersent1.pkl", 'wb') as f:
        f.write(file_url.content)
    file_url = requests.get("https://dl.fbaipublicfiles.com/infersent/infersent2.pkl")
    with open("infersent/infersent2.pkl", 'wb') as f:
        f.write(file_url.content)


if __name__ == "__main__":
    run()