import numpy as np
import pandas as pd
import os
import cv2
import torch
from torch.autograd import Variable
import random
import math
from skimage import transform

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def data_arguementaion(image, label):
    filp_p = np.random.rand()
    if filp_p > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)

    return image, label


def random_shift_scale_rotateN(images, shift_limit=(-0.0625,0.0625), scale_limit=(1/1.1,1.1),
                               rotate_limit=(-45,45), aspect_limit = (1,1),  borderMode=cv2.BORDER_REFLECT_101 , u=0.5):
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height,width,channel = images[0].shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        aspect = random.uniform(aspect_limit[0],aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        for n, image in enumerate(images):
            images[n] = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return images



def get_mini_batch_data_levelSet(data_path, img_name, img_size=256,gpu_num=1):

    img_path = os.path.join(data_path, 'train_data', img_name)
    label_path = os.path.join(data_path, 'train_label', img_name)

    LevelSetLabel_Name = img_name + '.npy'
    levelSet_Label_Path = os.path.join(data_path, 'Level_set_label', LevelSetLabel_Name)

    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    LevelSetLabel = np.load(levelSet_Label_Path)

    Final = LevelSetLabel/1000
    Final_1 = transform.resize(Final,(img_size, img_size))
    Final_LevelSet_Label = Final_1*1000

    #Normalization
    MaxDistance = np.max(Final_LevelSet_Label)
    MinDistance = np.min(Final_LevelSet_Label)

    Final_LevelSet_Label = Final_LevelSet_Label/MaxDistance


    img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

    # Data Augmentation:

    #img, label = random_shift_scale_rotateN([img,label])
    # flip_p = np.random.uniform(0, 1)
    # img = flip(img, flip_p)
    # label = flip(label, flip_p)
    # img = resize_img

    img = np.transpose(img, [2, 0, 1])

    tmp_gt = label.copy()
    label = np.transpose(label, [2, 0, 1])

    Final_LevelSet_Label = Final_LevelSet_Label[np.newaxis,:,:]
    img = Variable(torch.from_numpy(img)).float().cuda(gpu_num)
    img = torch.unsqueeze(img, 0)
    label = Variable(torch.from_numpy(label)).long().cuda(gpu_num)
    LevelSetLabel = Variable(torch.from_numpy(Final_LevelSet_Label)).float().cuda(gpu_num)

    return img, label,LevelSetLabel,tmp_gt

def get_mini_batch_data(data_path, img_name, img_size=256,gpu_num=1):

    img_path = os.path.join(data_path, 'train_data', img_name)
    label_path = os.path.join(data_path, 'train_label', img_name)

    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

    # Data Augmentation:

    #img, label = random_shift_scale_rotateN([img,label])
    # flip_p = np.random.uniform(0, 1)
    # img = flip(img, flip_p)
    # label = flip(label, flip_p)
    # img = resize_img

    img = np.transpose(img, [2, 0, 1])

    tmp_gt = label.copy()
    label = np.transpose(label, [2, 0, 1])

    img = Variable(torch.from_numpy(img)).float().cuda(gpu_num)
    img = torch.unsqueeze(img, 0)
    label = Variable(torch.from_numpy(label)).long().cuda(gpu_num)

    return img, label,tmp_gt

def get_data(data_path, img_name, img_size=256, n_classes=4,gpu_num=0):

    def get_label(label):
        tmp_gt = label.copy()
        label = np.transpose(label, [2, 0, 1])
        # label = Variable(torch.from_numpy(label)).cuda()
        label = Variable(torch.from_numpy(label)).long().cuda(gpu_num)
        return label,tmp_gt

    if n_classes==2:
        img_path = os.path.join(data_path,'4', 'train_data', img_name)
        label_path = os.path.join(data_path,'4', 'train_label', img_name)

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        new_label = np.zeros_like(label)
        new_label[label==3] = 1
        label = new_label
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float().cuda(gpu_num)
        img = torch.unsqueeze(img, 0)
        label, tmp_gt = get_label(label)
        return img, label, tmp_gt

    img_path = os.path.join(data_path,str(n_classes), 'train_data', img_name)
    label_path = os.path.join(data_path, str(n_classes),'train_label', img_name)

    img = cv2.imread(img_path)
    # print img_path
    label = cv2.imread(label_path)
    img, label = data_arguementaion(img, label)
    img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

    img = np.transpose(img, [2, 0, 1])
    img = Variable(torch.from_numpy(img)).float().cuda()
    img = torch.unsqueeze(img, 0)

    label, tmp_gt = get_label(label)

    return img, label, tmp_gt

def get_new_data(data_path, img_name, img_size=256,gpu_num=1):

    img_path = os.path.join(data_path, 'train_data', img_name)
    label_path = os.path.join(data_path, 'train_label', img_name)

    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
    label_1 = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
    label_2 = cv2.resize(label, (img_size/2, img_size/2), interpolation=cv2.INTER_AREA)[:, :, :1]
    label_3 = cv2.resize(label, (img_size/4, img_size/4), interpolation=cv2.INTER_AREA)[:, :, :1]
    label_4 = cv2.resize(label, (img_size/8, img_size/8), interpolation=cv2.INTER_AREA)[:, :, :1]
    label_5 = cv2.resize(label, (img_size/16, img_size/16), interpolation=cv2.INTER_AREA)[:, :, :1]
    label_6 = cv2.resize(label, (img_size / 32, img_size / 32), interpolation=cv2.INTER_AREA)[:, :, :1]


    # Data Augmentation:

    # img, label = random_shift_scale_rotateN([img,label])

    # flip_p = np.random.uniform(0, 1)
    # img = flip(img, flip_p)
    # label = flip(label, flip_p)
    # img = resize_img

    def get_label(label):
        tmp_gt = label.copy()
        label = np.transpose(label, [2, 0, 1])
        label = Variable(torch.from_numpy(label)).long().cuda(gpu_num)

        return label,tmp_gt

    img = np.transpose(img, [2, 0, 1])
    img = Variable(torch.from_numpy(img)).float().cuda(gpu_num)
    img = torch.unsqueeze(img, 0)

    label_1, tmp_gt_1 = get_label(label_1)
    label_2, tmp_gt_2 = get_label(label_2)
    label_3, tmp_gt_3 = get_label(label_3)
    label_4, tmp_gt_4 = get_label(label_4)
    label_5, tmp_gt_5 = get_label(label_5)
    label_6, tmp_gt_6 = get_label(label_6)

    label_list = [[label_1, tmp_gt_1], [label_2, tmp_gt_2], [label_3, tmp_gt_3], [label_4, tmp_gt_4],
                  [label_5, tmp_gt_5], [label_6, tmp_gt_6]]

    return img, label_list


def decode_pixel_label(acc_label):
    label_img = np.zeros([acc_label.shape[0], acc_label.shape[1], 3], dtype=np.uint8)
    for i in range(acc_label.shape[0]):
        for j in range(acc_label.shape[1]):
            if (acc_label[i,j]==1).all():
                label_img[i,j]=[255,255,0]
            elif (acc_label[i,j]==2).all():
                label_img[i,j]=[255,0,0]
            elif (acc_label[i,j]==3).all():
                label_img[i,j]=[0,0,255]

    return label_img

def img_addWeighted(ori_img,pred_img,pred_infor):
    left, right, top = pred_infor['position']
    w,h = pred_infor['size'][:2]
    ROI_img = ori_img[top:, left:right, :]
    pred_img = cv2.resize(pred_img, (h,w),interpolation=cv2.INTER_AREA)

    ROI_img = cv2.addWeighted(ROI_img,0.6,pred_img,0.4,0)
    ori_img[top:, left:right, :] = ROI_img
    return ori_img

def find_boundry(tmp_array):
    max = 0
    min = None
    for i in xrange(1,int(len(tmp_array))/2):
        left = tmp_array[:i]
        right = tmp_array[i:]
        diff = np.sum(right)-np.sum(left)
        if diff >=max:
            max = diff
            left_index = i
        left = tmp_array[:-i]
        right = tmp_array[-i:]
        diff = np.sum(left)-np.sum(right)
        if min is None or min<=diff:
            min = diff
            right_index = len(tmp_array)-i
    return left_index, right_index


def get_top_donw_boundry(ROI_img):
    index_array = np.zeros([ROI_img.shape[1], 2])
    for i in xrange(0,ROI_img.shape[1],30):
        tmp_array = ROI_img[:, i]
        if tmp_array.sum() < 20:
            continue
        left_index, right_index = find_boundry(tmp_array)
        index_array[i] = [left_index, right_index]
    # x = np.linspace(0, ROI_img.shape[0], ROI_img.shape[0])
    x = np.arange(ROI_img.shape[1])
    mask = index_array.sum(axis=1) != 0
    index_array = index_array[mask]
    x = x[mask]
    y1 = index_array[:, 0]
    y2 = index_array[:, 1]

    z1 = np.polyfit(x, y1, 2)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(0, ROI_img.shape[0] - 1, ROI_img.shape[0])
    plt_y_1 = np.polyval(p1, plt_x)

    z2 = np.polyfit(x, y2, 2)
    p2 = np.poly1d(z2)
    plt_y_2 = np.polyval(p2, plt_x)
    return plt_x, plt_y_1, plt_y_2

def crop_boundry(ROI_img, pred_img):
    plt_x, plt_y_1, plt_y_2 = get_top_donw_boundry(pred_img)
    y_sum = np.sum(pred_img, axis=1)
    mask = y_sum != 0
    y_ = np.arange(pred_img.shape[0])
    y_ = y_[mask]
    y_max = y_[-1]
    y_min = y_[0]
    for i in range(y_min,y_max+1):
        for j in range(pred_img.shape[1]):
            if pred_img[i,j]==1:
                ROI_img[i,j] = [255,255,0]
            elif pred_img[i,j]==2:
                ROI_img[i,j] = [255,0,255]
            elif pred_img[i,j] == 3:
                ROI_img[i,j] = [0,255,255]

    # plt_x, plt_y_1, plt_y_2 = get_top_donw_boundry(ROI_img)
    new_index_1 = np.stack([plt_x, plt_y_1], 1).astype(np.int32)
    new_index_2 = np.stack([plt_x, plt_y_2], 1).astype(np.int32)
    cv2.polylines(ROI_img, [new_index_1], False, [0,0,255], 4)
    cv2.polylines(ROI_img, [new_index_2], False, [0, 0, 255], 4)
    return ROI_img

def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)
    meanIU = np.mean(IU)
    pos[pos==0]=1
    res[res==0]=1
    pixelAccuracy = np.sum(tp) / np.sum(confusion)
    meanAccuracy = np.mean(tp / pos)
    classAccuracy = np.mean(tp / res)
    return  meanIU,pixelAccuracy,meanAccuracy,classAccuracy


def dice_loss(m1, m2, is_average=True):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    if is_average:
        score = scores.sum()/num
        return score
    else:
        return scores

def my_ployfit(x,y,num,start,end,ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(start, end, num)
    plt_y = np.polyval(p1, plt_x)
    return plt_x,plt_y

def process_csv(data):
    for key in data.keys():
        data[key] = data[key].astype(np.float32)
    # data = data.sort_values(by='0')
    r_sum = data.iloc[:, 1:].apply(lambda x: x.sum(), axis=1).values.astype(np.bool)
    data = data.iloc[r_sum, :]
    return data

def process_x_y(csv_data,idx,start,end,flag=False):
    # x = csv_data['0'].values * img.shape[1] / 16.0
    # y = csv_data[str(idx + 1)].values * img.shape[0] / 14.0
    x = csv_data['0'].values * 2130 / 16.0
    y = csv_data[str(idx + 1)].values * 1864 / 14.0
    mask = map(lambda i: not np.isnan(i), y)
    y = y[mask]
    x = x[mask]
    if flag:
        start = x[5]
        end = x[-5:-4]
    x, y = my_ployfit(x, y, num=2130 - 1, start=start, end=end)
    return x,y

def get_truth(ROI_img, img_name, ally, BeginY):
    ori_dir = '/home/intern1/guanghuixu/segmentation/data/eyes/'
    left_right = {'L': '1', 'R': '0'}
    tmp_name = img_name.split('_')[:2]
    img_id = img_name.split('_')[-1]
    img_id = int(img_id.split('.')[0])
    tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]]
    csv_dir = os.listdir(os.path.join(ori_dir, tmp_name))
    csv_path = [x for x in csv_dir if x.endswith('.csv')]
    csv_path = os.path.join(ori_dir, tmp_name, csv_path[0])
    csv_data = pd.read_csv(csv_path)

    Lens_front = csv_data[:800]
    Lens_back = csv_data[4015:4815]
    Lens1 = process_csv(Lens_front)
    Lens2 = process_csv(Lens_back)
    Lens2 = Lens2.sort_index(ascending=False)
    Lens = pd.concat([Lens1, Lens2])

    Cortex_front = csv_data[803:1603]
    Cortex_back = csv_data[3212:4012]
    Cortex1 = process_csv(Cortex_front)
    Cortex2 = process_csv(Cortex_back)
    Cortex2 = Cortex2.sort_index(ascending=False)
    Cortex = pd.concat([Cortex1, Cortex2])

    Nucleus_front = csv_data[1606:2406]
    Nucleus_back = csv_data[2409:3209]
    Nucleus1 = process_csv(Nucleus_front)
    Nucleus2 = process_csv(Nucleus_back)
    Nucleus2 = Nucleus2.sort_index(ascending=False)

    # for color_id, csv_data in enumerate([[Nucleus1,Nucleus2],[Lens1,Lens2],[Cortex1,Cortex2]]):
    #     front, back = csv_data
    #     front_x, front_y = process_x_y(front, img_id, start=ally[0], end=ally[1], flag=True)
    #     back_x, back_y = process_x_y(back, img_id, start=ally[1], end=ally[0], flag=True)
    #     x = np.stack([front_x, back_x]).reshape([-1])
    #     y = np.stack([front_y, back_y]).reshape([-1])
    #     # x = x - ally[0]
    #     # y = y - BeginY
    #     new_index = zip(x, y)
    #     new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    #     cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0), 4)

    for color_id, csv_data in enumerate([[Nucleus1, Nucleus2], [Lens1, Lens2], [Cortex1, Cortex2]]):
        front, back = csv_data
        front_x, front_y = process_x_y(front, img_id, start=ally[0], end=ally[1], flag=True)
        back_x, back_y = process_x_y(back, img_id, start=ally[1], end=ally[0], flag=True)
        x = np.stack([front_x, back_x]).reshape([-1])
        y = np.stack([front_y, back_y]).reshape([-1])
        # x = x - ally[0]
        # y = y - BeginY
        front_x = front_x.reshape([-1])
        front_y = front_y.reshape([-1])
        new_index = zip(front_x, front_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0), 4)

        back_x = back_x.reshape([-1])
        back_y = back_y.reshape([-1])
        new_index = zip(back_x, back_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0), 4)

    return ROI_img


def get_truth_annotation(img_name, ally):
    ori_dir = '/home/intern1/guanghuixu/segmentation/data/eyes/'
    left_right = {'L': '1', 'R': '0'}
    tmp_name = img_name.split('_')[:2]
    img_id = img_name.split('_')[-1]
    img_id = int(img_id.split('.')[0])
    tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]]
    csv_dir = os.listdir(os.path.join(ori_dir, tmp_name))
    csv_path = [x for x in csv_dir if x.endswith('.csv')]
    csv_path = os.path.join(ori_dir, tmp_name, csv_path[0])
    csv_data = pd.read_csv(csv_path)

    Lens_front = csv_data[:800]
    Lens_back = csv_data[4015:4815]
    Lens1 = process_csv(Lens_front)
    Lens2 = process_csv(Lens_back)
    Lens2 = Lens2.sort_index(ascending=False)
    Lens = pd.concat([Lens1, Lens2])

    Cortex_front = csv_data[803:1603]
    Cortex_back = csv_data[3212:4012]
    Cortex1 = process_csv(Cortex_front)
    Cortex2 = process_csv(Cortex_back)
    Cortex2 = Cortex2.sort_index(ascending=False)
    Cortex = pd.concat([Cortex1, Cortex2])

    Nucleus_front = csv_data[1606:2406]
    Nucleus_back = csv_data[2409:3209]
    Nucleus1 = process_csv(Nucleus_front)
    Nucleus2 = process_csv(Nucleus_back)
    Nucleus2 = Nucleus2.sort_index(ascending=False)

    # for color_id, csv_data in enumerate([[Nucleus1,Nucleus2],[Lens1,Lens2],[Cortex1,Cortex2]]):
    #     front, back = csv_data
    #     front_x, front_y = process_x_y(front, img_id, start=ally[0], end=ally[1], flag=True)
    #     back_x, back_y = process_x_y(back, img_id, start=ally[1], end=ally[0], flag=True)
    #     x = np.stack([front_x, back_x]).reshape([-1])
    #     y = np.stack([front_y, back_y]).reshape([-1])
    #     # x = x - ally[0]
    #     # y = y - BeginY
    #     new_index = zip(x, y)
    #     new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    #     cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0), 4)

    for color_id, csv_data in enumerate([[Nucleus1, Nucleus2]]):
        front, back = csv_data
        front_x, front_y = process_x_y(front, img_id, start=ally[0], end=ally[1], flag=True)
        back_x, back_y = process_x_y(back, img_id, start=ally[1], end=ally[0], flag=True)
        x = np.stack([front_x, back_x]).reshape([-1])
        y = np.stack([front_y, back_y]).reshape([-1])
        front_x = front_x.reshape([-1])
        front_y = front_y.reshape([-1])
        # new_index = zip(front_x, front_y)
        # new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])

        back_x = back_x.reshape([-1])
        back_y = back_y.reshape([-1])
        # new_index = zip(back_x, back_y)
        # new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    return front_x,front_y,back_x,back_y


def compute_loss(front_x, front_y, front_x_pred, front_y_pred):
    front_x = np.floor(front_x)
    front_y = np.floor(front_y)

    def new_index(x, y):
        index = np.argsort(x)
        x = x[index]
        y = y[index]
        x, index = np.unique(x, return_index=True)
        y = y[index]
        return x, y

    front_x, front_y = new_index(front_x, front_y)
    front_x_pred, front_y_pred = new_index(front_x_pred, front_y_pred)
    min_x = front_x[0]
    max_x = front_x[-1]
    if min_x < front_x_pred[0]:
        min_x = front_x_pred[0]
    if max_x > front_x_pred[-1]:
        max_x = front_x_pred[-1]


    left = np.where(front_x == min_x)[0][0]
    right = np.where(front_x == max_x)[0][0] + 1
    front_x = front_x[left:right]
    front_y = front_y[left:right]

    left = np.where(front_x_pred == min_x)[0][0]
    right = np.where(front_x_pred == max_x)[0][0] + 1
    front_x_pred = front_x_pred[left:right]
    front_y_pred = front_y_pred[left:right]

    ## my_ployfit
    front_x_pred, front_y_pred = my_ployfit(front_x_pred, front_y_pred, max_x - min_x + 1, min_x, max_x)

    loss = np.mean(np.abs(front_y - front_y_pred))
    return loss, front_x, front_y, front_x_pred, front_y_pred

def compute_MSE_pixel(FullImage, img_name, boundry_x_y, ally):
    front_x, front_y, back_x, back_y = get_truth_annotation(img_name, ally)
    x = boundry_x_y[1]
    y = boundry_x_y[0]
    mean_value = (y.max() - y.min()) / 2 + y.min()
    mask = y < mean_value
    front_y_pred = y[mask]
    front_x_pred = x[mask]
    mask = y >= mean_value
    back_y_pred = y[mask]
    back_x_pred = x[mask]

    front_loss, front_x, front_y, front_x_pred, front_y_pred = compute_loss(front_x, front_y, front_x_pred,
                                                                            front_y_pred)
    back_loss, back_x, back_y, back_x_pred, back_y_pred = compute_loss(back_x, back_y, back_x_pred, back_y_pred)



    new_index = zip(front_x_pred, front_y_pred)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    new_index = zip(front_x, front_y)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    new_index = zip(back_x_pred, back_y_pred)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    new_index = zip(back_x, back_y)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    cv2.imwrite('./MSE_pixel/%s'%img_name, FullImage)
    return front_loss, back_loss

def get_boundry_box(boundry):
    w = boundry.shape[1]
    center_x = w / 2
    top_idx = None
    down_idx = None
    new_y = boundry[:, center_x]
    center_y = new_y.shape[0] / 2
    for i in range(1, center_y):
        if new_y[center_y - i]:
            top_idx = center_y - i
            break
    for i in range(1, center_y):
        if new_y[center_y + i]:
            down_idx = center_y + i
            break

    # tmp_y = [top_idx - 100, down_idx + 100]
    # top = img[0:top_idx]
    # center = img[top_idx - 100:down_idx + 100]
    # bottom = img[down_idx:]
    return top_idx, down_idx, center_x
