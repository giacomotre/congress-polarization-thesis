the original dataset can be download at this link: https://data.stanford.edu/congress_text
It has been used in compliance with the Data Use Agreement and the ODC-BY 1.0 license

to perform the whole experiment follow these stwps:
- create the merged dataset with the ED python notebook file
- clean the speeches with spacy_cleaner_gpu.py
- build the fixed vocabulary for the tfidf vectorizer with vocabulary_builder notebook
- run the models (cuml_models.py and new_svm.py)
- run the explainability file
- run the cross_year_avaluation.py

the enviroment file has been created for the Tilburg Unvierstity facilieties 
hardware specification: A40 Nvidia GPU, Cuda compilation tools, release 11.8, V11.8.89

for the spacy_cleaner_gpu notbook the other environment has been used.




