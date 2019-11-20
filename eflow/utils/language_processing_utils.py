import nltk
from nltk.corpus import wordnet


def get_synonyms_antonyms(word):

    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return {word.replace("_", " ") for word in synonyms}, \
           {word.replace("_", " ") for word in antonyms}


def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    return {word.replace("_", " ") for word in synonyms}

def get_antonyms(word):
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return {word.replace("_", " ") for word in antonyms}