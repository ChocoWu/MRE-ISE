# Textual Scene Graph Parser


## Preparation
This code refers to [SPICE](https://github.com/peteanderson80/SPICE), , so you can refer [SPICE/README.md](https://github.com/peteanderson80/SPICE/blob/master/README.md) to prepare the requirements.
### Requirements
- java 1.8.0+
- Stanford [CoreNLP](http://stanfordnlp.github.io/CoreNLP/) 3.6.0
- Stanford [Scene Graph Parser](http://nlp.stanford.edu/software/scenegraph-parser.shtml)
- [Meteor](http://www.cs.cmu.edu/~alavie/METEOR/) 1.5 (for synset matching)

### Build
This zip file contains the pre-built SPICE-1.0.jar and all libraries required to run it, except for Stanford CoreNLP.
Therefore, you can run the following command:
```angular2html
$ bash ./get_stanford_models.sh
```
or manually download the CoreNLP 3.6.0 code and models jar files into [lib](lib).
Instructions for using SPICE can be found in [SPICE/README.md](https://github.com/peteanderson80/SPICE/blob/master/README.md), and the `spice.1.0.jar` can be downloaded in this [website](https://panderson.me/spice/).

## TSG parsing

Firstly, modified the following variable in the [textual_scene_graph.py](textual_scene_graph.py):
```angular2html
    FILE_DIR = '../../data/txt/' # the directory that storages the original annotation file
    INPUT_TMP_DIR = '../../data/spice/input/'  # the temporary directory that storages the temp input file for SPICE
    OUTPUT_DIR = '../../data/spice/output/'  # the temporary directory that storages the temp output file for SPICE
    DIST_DIT = '../../data/tsg/'  # the final directory that storages the parsed results
```
Then, run the command:
```angular2html
python textual_scene_graph.py
```

The final file structure:
```angular2html
[
    {
        'token': ['RT', '@DenisLaw_WFT', ':', 'New', 'breed', 'of', 'Crocodile', 'discovered', 'in', 'South', 'Wales', 'woodland'], 
        'h': {'name': 'Crocodile', 'pos': [6, 7]}, 
        't': {'name': 'South Wales', 'pos': [9, 11]}, 
        'img_id': 'twitter_stream_2018_10_10_13_0_2_142.jpg', 
        'relation': '/misc/loc/held_on',
        'TSG': {
            'obj': ['breed'],
            'attr': ['breed', 'New'],
            'rel': ['breed', 'of', 'Crocodile']
        }
    },
    ...
]
```
