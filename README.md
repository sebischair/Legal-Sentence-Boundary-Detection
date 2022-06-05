# Legal-Sentence-Boundary-Detection

Sentence boundary detection is a critical task in many natural language processing modules and methods. 
Previously a golden standard was assumed but when looking at specific domains (such as the legal domain), existing methods are not performing well.

The aim of this project was to investigate the applicability of existing solutions and the experiment with new methods in the legal domain.

The code is provided as is and ranges from the preprocessing modules to the implementation of the specific methods. Be aware that this code is old and not necessarily every piece of code will immediately be runnable. The project was written with Python 3.7, but a newer version will (most likely) work as well.

Third-party dependencies are:
* nltk
* PyTorch
* gensim (Code needs to be changed when using the 4.x release of gensim)
* pycrfsuite

# Code

The code should be executed from the `src` folder. You might need to change specific paths depending on your setup.

For most method there are two associated files: `[...]Trainer.py` which defines all the training procedures and `[...]Train.py` which is a simple script to start the training. This code includes implementations for Conditial Random Fields, (different) Neural Networks, NLTK, OpenNLP and some heuristic/rule-based approaches for sentence boundary detection. To start the training procedure of a model class execute `[...]Train.py`, which will save the model to a `Models` directory in the root directory.

There also is a very simple annotation tool in this repository: `Annotator.py`. It is in a very early development stage and not every functionality was finalized. Expanding the annotation tool to other annotation tasks should be rather easy.

If you want to test OpenNLP you need a working corenlp version. See `src/Modules/OpenNLPModule` for more information.

# Data

Contains the annotated data in a JSON format. Each JSON file contains a `Annotations` section with a list of annotations. Each annotation will includes the last word in a sentence with its start and end index in characters.
The text of the document (referenced by the indices) is given in the `Text` field. 

The file prefix denotes the document type:
* `jug_`: German judgments or verdicts
* `ges_`: German laws
* `prv_`: Privacy policies
* `agb_`: Terms of Service
* `wiki_`: Wikipedia documents