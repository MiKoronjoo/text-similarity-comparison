from typing import List, Dict, Tuple

import spacy
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_wiki_page_text

nlp = spacy.load('en_core_web_sm')


def read_file(file_path: str):
    with open(file_path) as fp:
        text = fp.read().replace('\n', ' ')
    return text


def preprocess(text: str):
    stop_words = set(stopwords.words('english'))

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words and lower case
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]

    # stemming
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # POS tagging
    pos_tags = pos_tag(tokens)

    # NER
    doc = nlp(text)
    named_entities = [(X.text, X.label_) for X in doc.ents]

    return tokens, pos_tags, named_entities


def process_verbs(text: str):
    verbs = [word for word in text if wn.synsets(word, 'v')]
    synonyms = {verb: wn.synsets(verb, 'v')[0].lemmas()[0].name() for verb in verbs}
    hyp_hyponyms = {verb: (wn.synsets(verb, 'v')[0].hypernyms(),
                           wn.synsets(verb, 'v')[0].hyponyms()) for verb in verbs}
    return synonyms, hyp_hyponyms


def replace_with_synonyms(tokens: List[str], syns: Dict[str, str]):
    return [syns.get(word, word) for word in tokens]


def replace_with_hypernyms(
        tokens: List[str],
        pos: List[Tuple[str, str]],
        hyp_hyponyms: Dict[str, tuple],
        named_entities: List[Tuple[str, str]]
):
    replaced = []
    for word, tag in zip(tokens, pos):
        if word in named_entities:
            replaced.append(word)
        elif hyp_hyponyms.get(word) and hyp_hyponyms[word][0]:
            replaced.append(hyp_hyponyms[word][0][0].lemmas()[0].name())
        else:
            replaced.append(word + "_" + tag[1])
    return replaced


def compute_similarity(text1: str, text2: str):
    documents = [text1, text2]
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
    return cosine_similarity(sparse_matrix[0:1], sparse_matrix[1:2])


if __name__ == '__main__':
    get_wiki_page_text('Lemmatisation', write_to_file=True)
    get_wiki_page_text('Stemming', write_to_file=True)
    file1 = read_file('Lemmatisation')
    file2 = read_file('Stemming')

    tokens1, pos_tags1, named_entities1 = preprocess(file1)
    tokens2, pos_tags2, named_entities2 = preprocess(file2)

    syns1, hyp_hyponyms1 = process_verbs(tokens1)
    syns2, hyp_hyponyms2 = process_verbs(tokens2)

    tokens1 = replace_with_synonyms(tokens1, syns1)
    tokens2 = replace_with_synonyms(tokens2, syns2)

    tokens1 = replace_with_hypernyms(tokens1, pos_tags1, hyp_hyponyms1, named_entities1)
    tokens2 = replace_with_hypernyms(tokens2, pos_tags2, hyp_hyponyms2, named_entities2)

    similarity = compute_similarity(' '.join(tokens1), ' '.join(tokens2))

    print(f'Similarity between the two files: % {100 * similarity[0][0]:.6f}')
