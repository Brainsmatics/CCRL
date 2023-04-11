import cv2
import glob
import torch.utils.data as data
import numpy as np
import torch
import os

def get_imgInput(data):
    print('-'*30)
    print('Loading images...')
    print('-'*30)

    train_image_list = []
    train_label_list = []
    val_image_list = []
    val_label_list = []
    test_image_list = []
    test_label_list = []

    for filename in data["traindata"]:
        img = cv2.imread(filename)
        train_image_list.append(img)
    for filename in data["trainlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        train_label_list.append(img)
    for filename in data["valdata"]:
        img = cv2.imread(filename)
        val_image_list.append(img)
    for filename in data["vallabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        val_label_list.append(img)
    for filename in data["testdata"]:
        img = cv2.imread(filename)
        test_image_list.append(img)
    for filename in data["testlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        test_label_list.append(img)


    return train_image_list, train_label_list, val_image_list, val_label_list, test_image_list, test_label_list

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, target):
        out1, _ = self.transform(inp, target)

        return out1, out1


class TransformRot:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, target):
        out1, _ = self.transform(inp, target)
        # out2, _ = self.transform(inp, target)
        return out1, out1


def get_imgInput_dataset(root  #"./data/imgInput/"
                           , num_labels, transform_train=None, transform_val=None, transform_forsemi=None):

    path_train_data = os.listdir(root + 'myTraining_Data')
    path_train_data = [root+'myTraining_Data/'+i for i in path_train_data]
    path_valid_data = os.listdir(root + r'myValid_Data')
    path_valid_data = [root + 'myValid_Data/' + i for i in path_valid_data]
    path_test_data = os.listdir(root + r'myTest_Data')
    path_test_data = [root + 'myTest_Data/' + i for i in path_test_data]
    #  fix load files seq
    path_train_data.sort()
    path_valid_data.sort()
    path_test_data.sort()

    #  index of fixed labeled data
    if num_labels < 3000:
        a = np.loadtxt(r"data_id/img_id"+str(num_labels)+".txt", dtype='str')
        a = [root + r"myTraining_Data/" + item for item in a]
        train_labeled_idxs = [path_train_data.index(item) for item in a]
        train_unlabeled_idxs = list(set(list(range(len(path_train_data)))) - set(train_labeled_idxs))
    else:
        a = np.loadtxt(r"data_id/img_id" + str(num_labels) + ".txt", dtype='str')
        a = [root + r"myTraining_Data/" + item for item in a]
        train_labeled_idxs = [path_train_data.index(item) for item in a]
        train_unlabeled_idxs = []   
        # print(train_labeled_idxs)
    # label 的路径
    path_train_label = ['/'.join(item.replace("myTraining_Data", "myTraining_Label").split("/")[:-1]) +"/"+
                        item.split("/")[-1][:-4]+".png" for item
                        in path_train_data]  
    path_valid_label = ['/'.join(item.replace("myValid_Data", "myValid_Label").split("/")[:-1]) +"/"+
                        item.split("/")[-1][:-4]+".png" for item
                        in path_valid_data]
    path_test_label = ['/'.join(item.replace("myTest_Data", "myTest_Label").split("/")[:-1]) +"/"+
                        item.split("/")[-1][:-4]+".png" for item
                        in path_test_data]

    data = {"traindata": path_train_data,
            "trainlabel": path_train_label,
            "valdata": path_valid_data,
            "vallabel": path_valid_label,
            "testdata": path_test_data,
            "testlabel": path_test_label}

    # load data
    train_data, train_label, val_data, val_label, test_data, test_label = get_imgInput(data)

    val_name = path_valid_data
    test_name= path_test_data
    train_name = path_train_data


    train_labeled_dataset = imgInput_labeled(train_data, train_label,name=train_name,indexs=train_labeled_idxs,
                                               transform=transform_train)
    train_unlabeled_dataset = imgInput_unlabeled(train_data, train_label, indexs=train_unlabeled_idxs,
                                                   transform=TransformTwice(transform_train))
    val_dataset = imgInput_labeled(val_data, val_label, name=val_name,  indexs=None, transform=transform_val)
    test_dataset = imgInput_labeled(test_data, test_label, name=test_name, indexs=None, transform=transform_val)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_data)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255




def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class imgInput_labeled(data.Dataset):

    def __init__(self, data, label, name = None, indexs=None,
                 transform=None):

        self.data = data
        self.targets = label
        self.transform = transform
        self.name = name


        if indexs is not None: 
            self.data = [self.data[item] for item in indexs]
            self.targets = [self.targets[item] for item in indexs]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.name is not None:
            return img, target, self.name[index]
        else:

            return img, target



    def __len__(self):
        return len(self.data)



class imgInput_unlabeled(data.Dataset):

    def __init__(self, data, label, indexs=None,
                 transform=None):

        self.data = data
        self.targets = [-1*np.ones_like(label[item]) for item in range(0,len(label))] 
        

        self.transform = transform

        if indexs is not None:
            self.data = [self.data[item] for item in indexs]
            self.targets = [self.targets[item] for item in indexs]



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.data)


class imgInput_test(data.Dataset):

    def __init__(self, data,label,transform=None):

        self.data = data
        self.targets = [-1*np.ones_like(label[item]) for item in range(0,len(label))] 
        self.name = [-1*np.ones_like(label[item]) for item in range(0,len(label))]
        self.transform = transform



    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target,self.name[index]

    def __len__(self):
        return len(self.data)

def get_imgInput_test(img_path,label_path,transform_val=None):
    print('-'*30)
    print('Loading images...')
    print('-'*30)
    test_image_list = []
    test_label_list= []
    for filename in os.listdir(label_path):
        img = cv2.imread(label_path+r'/'+filename, cv2.IMREAD_GRAYSCALE)
        test_label_list.append(img)
    for filename in os.listdir(img_path):
        img = cv2.imread(img_path+r'/'+filename)
        test_image_list.append(img)
    return imgInput_test(test_image_list,test_label_list,transform=TransformTwice(transform_val))


def get_preTrain(img_path,label_path,transform=None):
    print('-'*30)
    print('Loading images...')
    print('-'*30)
    test_image_list = []
    test_label_list= []
    for filename in os.listdir(label_path):
        img = cv2.imread(label_path+r'/'+filename,cv2.IMREAD_GRAYSCALE)
        test_label_list.append(img)
    for filename in os.listdir(img_path):
        img = cv2.imread(img_path+r'/'+filename,cv2.IMREAD_GRAYSCALE)
        test_image_list.append(img)
    return pretrain_test(test_image_list,test_label_list,transform)

class pretrain_test(data.Dataset):

    def __init__(self, data,label,transform):

        self.data = data
        self.targets =data
        self.transform = transform
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # img=np.transpose(img, (2,1, 0))
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.data)