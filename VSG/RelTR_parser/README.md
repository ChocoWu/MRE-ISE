# Visual Scene Graph Parser


## Preparation
We employ the [RelTR](https://github.com/yrcong/RelTR) to parse the visual scene graph.
Firstly, download the code:
```angular2html
git clone https://github.com/yrcong/RelTR.git
```
The folder structure of the dataset is shown below:
```angular2html
|RelTR_parser
|── models
|── utils
|── visual_scene_graph.py
|── ...
```
Please refer the requirements in the [RelTR](https://github.com/yrcong/RelTR).


## VSG parsing
Firstly, modified the following variable in the [visual_scene_graph.py](visual_scene_graph.py):
```angular2html
    FILE_DIR = '../data/tsg/'  # the directory that storages the tsg data.
    IMG_DIR = '../data/img_org/'  # the directory that storages the original image file
    DIST_DIR = '../data/vsg_tsg/'  # the final directory that storages the parsed results
```
Then, run the command:
```angular2html
python visual_scene_graph.py
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
            'obj': [['breed'], ...],
            'attr': [['breed', 'New'], ...],
            'rel': [['breed', 'of', 'Crocodile'], ...]
        },
        'VSG': {
            'bbox': [[0, 0, 800, 600], [70, 209, 190, 247], ...],
            'bbox_attri': ["img", "wing", "wing", "bird", ...],
            'rel': [
                {"s_index": 1, "o_index": 2, "name": "of"},
                ...
            ]
        }
    },
    ...
]
```

