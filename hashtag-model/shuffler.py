import os
import numpy as np

from tqdm import tqdm

cwd = os.getcwd()

datafilename = os.path.join(cwd, 'data_list.txt')
tagfilename = os.path.join(cwd, 'tag_list.txt')
datafileshufname = os.path.join(cwd, 'data_list_shuf.txt')
tagfileshufname = os.path.join(cwd, 'tag_list_shuf.txt')

with open(datafilename, 'r') as datafile:
    with open(tagfilename, 'r') as tagfile:
        with open(datafileshufname, 'w') as datafileshuf:
            with open(tagfileshufname, 'w') as tagfileshuf:
                datalines = datafile.readlines()
                taglines = tagfile.readlines()
                data = np.array([(d, t) for d, t in zip(datalines, taglines)])
                np.random.shuffle(data)
                for d, t in tqdm(data):
                    datafileshuf.write(d)
                    tagfileshuf.write(t)
