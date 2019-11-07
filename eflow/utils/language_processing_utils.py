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

    return set(synonyms), set(antonyms)


def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    return set(synonyms)




def get_antonyms(word):
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return set(antonyms)