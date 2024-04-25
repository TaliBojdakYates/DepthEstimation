import shutil
import  os

train_root = 'C:/Users/talib/.keras/datasets/nyudepthv2/train/'

train_names = []
for main_folder in os.listdir(train_root):
    outer_dir = train_root + main_folder
    for inner_file in os.listdir(outer_dir):
        train_names.append(outer_dir + '/' + inner_file)

for i in train_names[:1000]:
    try:
        s = i.split('/')
        path = s[-2] + '_' + s[-1]
        destination = 'test/' + path
        shutil.move(i, destination)
    except Exception as e:
        print(e)