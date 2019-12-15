import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
from scipy.ndimage import rotate
import random

def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


def flip_image(image, axis):
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)
    return new_img_like(image, data=new_data)

def random_boolean():
    return np.random.choice([True, False])

#添加高斯噪声
def add_gauss_noise(image, u=0,deta=[0.3+i*0.1 for i in range(0,5)]):
    new_data = np.copy(image.get_data())
    new_data += np.random.normal(u,random.sample(deta,1)[0],new_data.shape)
    return new_img_like(image, data=new_data)

#xy方向的旋转【90，180，270】
def rotate_xy(image,axis=(1,0),theta=[0,90,180,270],isseg=True):
    order = 0 if isseg == True else 5
    new_data = np.copy(image.get_data())
    new_data = rotate(new_data,random.sample(theta,1)[0],reshape=False, order=order, mode='nearest')
    return new_img_like(image,data=new_data)

#随机产生翻转的axis_list
def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis

#随机产生缩放因子，与数据维数相同，3维
def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)

def distort_image(image, flip_axis=None, scale_factor=None,rotation=False,isseg=False,add_gauss=False):
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    if rotation:
        image = rotate_xy(image,isseg=isseg)
    if add_gauss:
        image = add_gauss_noise(image)
    return image

def augment_data(data, truth, affine, scale_deviation=None, flip=False, add_gauss=False, rotation=False):
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    image = get_image(data, affine)
    img_data = resample_to_img(distort_image(image, flip_axis=flip_axis,
                                                    scale_factor=scale_factor,rotation=rotation,isseg=False,add_gauss=add_gauss), image,
                                        interpolation="continuous").get_data()
    truth_image = get_image(truth, affine)
    truth_data = resample_to_img(distort_image(truth_image, flip_axis=flip_axis, scale_factor=scale_factor,rotation=rotation,isseg=True,add_gauss=False),
                                 truth_image, interpolation="nearest").get_data()
    return img_data, truth_data

def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)