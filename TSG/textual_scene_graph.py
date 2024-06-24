import os
import json
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile
import re


def load_data(filename):
    res = []
    with open(filename, mode='r') as f:
        for line in f:
            json_line = ast.literal_eval(line)
            res.append(json_line)
    return res


def sparse_textual_scene_graph(data, tmp_dir, out_dir, data_mode='train'):
    """
    to parse textual scene graph by using SPICE, see more information in https://github.com/peteanderson80/SPICE
    :param data:
    :return:
    """
    # Prepare temp input file for the SPICE scorer
    input_data = []
    for id, instance in enumerate(data):
        _temp = ' '.join(instance['token'])
        img_id = instance['img_id']
        input_data.append({
            "image_id": img_id,
            "test": _temp,
            "refs": [_temp]
        })

    cwd = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(cwd, tmp_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, mode='w')
    json.dump(input_data, in_file, indent=2)
    in_file.close()

    # cwd = os.path.dirname(os.path.abspath(__file__))
    # temp_dir = os.path.join(cwd, tmp_dir)
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    # in_file = os.path.join(tmp_dir, data_mode)
    # with open(in_file, mode='w') as f:
    #     json.dump(input_data, f, indent=2)

    # Start job
    SPICE_JAR = 'spice-1.0.jar'
    temp_dir = os.path.join(cwd, out_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    out_file = os.path.join(temp_dir, data_mode+'.json')
    spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
                 '-out', out_file,
                 '-detailed',
                 '-subset',
                 '-silent'
                 ]
    subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    # Read and process results
    with open(out_file, mode='r') as data_file:
        results = json.load(data_file)
    return results


def get_index(data, target_name):
    start_token = target_name.split()[0]
    end_token = target_name.split()[-1]
    start_index = data.index(start_token)
    end_index = data.index(end_token)
    return start_index, end_index


def combine(meta_data, tuple_data, target_file):
    """
    combine scene graph and meta data
    :param meta_data: the original data
    :param tuple_data: the parsed SG data
    :param target_file: the target file which saves the final scene graph
    :return:
    """

    assert len(tuple_data) == len(meta_data)
    for sg, md in zip(tuple_data, meta_data):
        # md['tuples'] = sg['test_tuples']
        _temp_sg = sg['test_tuples']
        obj = []
        attr = []
        relation = []
        for i in _temp_sg:
            if len(i['tuple']) == 1:
                # object
                obj.append(i['tuple'])
            elif len(i['tuple']) == 2:
                # attributes
                attr.append(i['tuple'])
            elif len(i['tuple']) == 3:
                relation.append(i['tuple'])
            else:
                raise EOFError('no SG obtained')
        md['TSG'] = {'obj': set(obj), 'attr': set(attr), 'rel': set(relation)}

    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f)


if __name__ == '__main__':
    print('parsing textual scene graph')
    FILE_DIR = '../data/txt/'
    INPUT_TMP_DIR = '../data/spice/input/'
    OUTPUT_DIR = '../data/spice/output/'
    DIST_DIR = '../data/tsg/'

    for i in ['ours_train.txt', 'ours_val.txt', 'ours_test.txt']:
        print(f'parsing {i} ... ')
        base_name = os.path.basename(i).split('.')[0]
        data = load_data(os.path.join(FILE_DIR, i))
        sg_data = sparse_textual_scene_graph(data, tmp_dir=INPUT_TMP_DIR, out_dir=OUTPUT_DIR,
                                             data_mode=base_name)
        combine(data, sg_data, os.path.join(DIST_DIR, f'{base_name}.json'))

    # print('parsing train data ... ')
    # train_data = load_data(os.path.join(FILE_DIR, 'ours_train.txt'))
    # train_tuple_data = sparse_textual_scene_graph(train_data, tmp_dir=INPUT_TMP_DIR, out_dir=OUTPUT_DIR, data_mode='train')
    # combine(train_data, train_tuple_data, os.path.join(DIST_DIT, 'train.json'))
    #
    # print('parsing valid data ... ')
    # vaild_data = load_data(os.path.join(FILE_DIR, 'ours_val.txt'))
    # valid_tuple_data = sparse_textual_scene_graph(train_data, tmp_dir=INPUT_TMP_DIR, out_dir=OUTPUT_DIR, data_mode='vaild')
    # combine(vaild_data, valid_tuple_data, os.path.join(DIST_DIT, 'val.json'))
    #
    # print('parsing test data ... ')
    # test_data = load_data(os.path.join(FILE_DIR, 'ours_test.txt'))
    # test_tuple_data = sparse_textual_scene_graph(train_data, tmp_dir=INPUT_TMP_DIR, out_dir=OUTPUT_DIR, data_mode='test')
    # combine(test_data, test_tuple_data, os.path.join(DIST_DIT, 'test.json'))








