# -*- coding: utf-8 -*-

import gzip
import glob


def load(folder, number_of_files=None, verbose=False):
    """
    Loads the sentences contained in the first n zipped files of a folder.
    The final point of the sentence is remove and the sentences are transformed into lowercase.
    If ``number_of_files`` is ``None``, then it lists all the files of the folder.
    :param folder: the folder to load the data from.
    :param number_of_files: the number of files to load.
    :param verbose: True if the method should print execution details. By default False.
    :return: a tuple containing (the list of lemmatized sentences, the list of original sentences)
    """
    files = list_files(folder, number_of_files)

    lemmatized_sentences = []
    original_sentences = []
    for zipped_file in files:
        if verbose:
            print("Parsing " + zipped_file)
        file = unzip(zipped_file)
        (lemmas, originals) = parse(file)
        original_sentences += originals
        lemmatized_sentences += lemmas

    return lemmatized_sentences, original_sentences


def list_files(folder, number_of_files=None):
    """
    Returns the path of .gz files in a folder.
    :param folder: The folder
    :param number_of_files: The number of files to list or ``None`` to list all.
    :return: The list of the files' path
    """
    paths = glob.glob(folder + "/*.gz")

    if number_of_files is None:
        return paths
    else:
        return paths[:number_of_files]


def is_end_of_sentence(lemme, word):
    return (lemme == "." and word == ".") or (lemme == "!" and word == "!") or (lemme == "?" and word == "?")


def parse(file):
    """
    Parse a file into a tuple: (list of lemmatized sentences, list of original sentences). All the sentences
    are converted into lower case and final point of the phrase is removed.
    :param file: the file to parse
    :return: the tuple
    """
    lines = file.splitlines()

    lemmatized_sentences = []
    original_sentences = []

    lemmatized_sentence = ""
    original_sentence = ""

    for line in lines:
        if line.startswith('#begin') or line.startswith('#end'):
            continue
            
        # delete all characteres non ascii
        line=line.encode("ascii", errors="ignore").decode()

        split = line.split()
        if len(split) != 2:
            continue

        word, lemma = split

        if is_end_of_sentence(lemma, word) and len(original_sentence) > 0:
            original_sentences.append(original_sentence.rstrip())
            lemmatized_sentences.append(lemmatized_sentence.rstrip())
            original_sentence = ""
            lemmatized_sentence = ""
        else:
            original_sentence += word.lower() + ' '
            lemmatized_sentence += lemma.lower() + ' '

    return lemmatized_sentences, original_sentences


def unzip(zipped_file):
    """
    Unzip a file and returns it in form of a list of lines.
    :param zipped_file: the path of the zipped file
    """
    with gzip.open(zipped_file, 'rt', encoding='ISO-8859-1') as file:
        file = file.read()
    return file
