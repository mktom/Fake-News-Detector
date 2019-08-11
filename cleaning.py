import pandas as pd
import spacy

def load_body_sentence(file):
    nlp = spacy.load('en_core_web_lg')
    df=pd.read_csv(file)
    ###df=pd.read_csv('train_bodies.csv')
    body_sentence = {}
    for row in df.iterrows():
        if row[1]["Body ID"] not in body_sentence:
            line = row[1]["articleBody"]
            nlpDoc = nlp(line)
            temp = [(" ".join([token.lemma_ for token in sent if not \
                               token.is_punct and not token.is_stop and \
                               not token.is_space])).lower() for sent in nlpDoc.sents]
            temp = ','.join([x for x in temp if x != ''])
            body_sentence[row[1]["Body ID"]] = temp
    return body_sentence
        




