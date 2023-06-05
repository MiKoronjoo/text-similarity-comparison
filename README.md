# Text Similarity Comparison Using Python NLP Tools
This project provides a simple Python script to compare the similarity of two text files, particularly two Wikipedia articles. The comparison process involves pre-processing, finding verb relations using WordNet, and computing cosine similarity.

## Getting Started
### Prerequisites
* Python 3.7 or later
* pip (Python Package Installer)

### Installation
First, clone this repository to your local machine using:
```sh
git clone https://github.com/MiKoronjoo/text-similarity-comparison
```
Next, navigate to the cloned repository:
```sh
cd text-similarity-comparison
```
Then, install the required Python packages:
```sh
pip install -r requirements.txt
```
You will also need to download the 'en_core_web_sm' model for spaCy:
```sh
python -m spacy download en_core_web_sm
```
You may need to download some resources for NLTK as well, which can be done by running:
```sh
python install_nltk_resources.py
```
### Running the Program
After installing the prerequisites, you can run the script by calling:
```sh
python text_similarity.py
```
## Description
This project applies basic Natural Language Processing (NLP) techniques to compute the similarity between two text files. It begins by pre-processing the text: removing stop words, tokenizing, stemming, and performing POS tagging and Named Entity Recognition (NER).

Next, using WordNet, the script identifies verbs in the text and finds their synonyms, hypernyms, and hyponyms. Finally, it computes the cosine similarity between the two processed text files, providing a measure of their similarity.