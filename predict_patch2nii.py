import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import nibabel as nib
import numpy as np

from training import load_old_model

def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[:-1]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = pad_args.tolist() + [[0, 0]] * (len(data.shape) - pad_args.shape[0])
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                      start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[:-1]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2],...]

def compute_patch_indices(image_shape, patch_shape, overlap, start=None):
    image_shape=np.asarray(image_shape)
    patch_shape=np.asarray(patch_shape)
    if isinstance(overlap, list):
        overlap = np.asarray([overlap])
    if start is None:
        n_patches = np.ceil(image_shape / (patch_shape - overlap))
        overflow = (patch_shape - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_shape - overlap
    return get_set_of_patch_indices(start, stop, step)

def reconstruct_from_patches(patches, patch_indices, data_shape, default_value=0):
    """
    Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
    patches are averaged.
    :param patches: List of numpy array patches.
    :param patch_indices: List of indices that corresponds to the list of patches.
    :param data_shape: Shape of the array from which the patches were extracted.
    :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
    be overwritten.
    :return: numpy array containing the data reconstructed by the patches.
    """
    data = np.ones(data_shape) * default_value
    image_shape = data_shape[:-1]
    count = np.zeros(data_shape, dtype=np.int)
    for patch, index in zip(patches, patch_indices):
        image_patch_shape = patch.shape[:-1]
        if np.any(index < 0):
            fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
            patch = patch[fix_patch[0]:,fix_patch[1]:,fix_patch[2]:,:]
            index[index < 0] = 0
        if np.any((index + image_patch_shape) >= image_shape):
            fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)* ((index + image_patch_shape) - image_shape)),
                                   dtype=np.int)
            patch = patch[:fix_patch[0], :fix_patch[1], :fix_patch[2],:]
        patch_index = np.zeros(data_shape, dtype=np.bool)
        patch_index[index[0]:index[0]+patch.shape[0],
                    index[1]:index[1]+patch.shape[1],
                    index[2]:index[2]+patch.shape[2],:] = True
        patch_data = np.zeros(data_shape)
        patch_data[patch_index] = patch.flatten()

        new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
        data[new_data_index] = patch_data[new_data_index]

        averaged_data_index = np.logical_and(patch_index, count > 0)
        if np.any(averaged_data_index):
            data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
        count[patch_index] += 1
    return data

def patch_wise_prediction(model, data, overlap, batch_size=8):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[1:-1]])
    predictions = list()
    indices = compute_patch_indices(data.shape[:-1], patch_shape=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size and i<len(indices):
            patch = get_patch_from_3d_data(data, patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        batch = np.asarray(batch)
        prediction = model.predict(batch,batch_size=batch.shape[0])
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = list(data.shape[:-1]) + [int(model.output.shape[-1])]
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)

def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[-1]):
        prediction_images.append(nib.Nifti1Image(prediction[0,:,:,:,i], affine))
    return prediction_images

def get_prediction_labels(prediction, threshold=0.5, labels=None):
    label_arrays = []
    label_data = np.argmax(prediction, axis=-1) + 1
    label_data[np.max(prediction, axis=-1) < threshold] = 0
    if labels:
        for value in np.unique(label_data).tolist()[1:]:
            label_data[label_data == value] = labels[value - 1]
    label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays

def prediction_to_image(prediction, affine,hdr, label_map, threshold=0.5, labels=None):
    '''if prediction.shape[-1] == 1:
        data = prediction[:,:,:,0]
        if label_map:
            label_map_data = np.zeros(prediction[:,:,:,0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
            #print(data)
    elif prediction.shape[-1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))'''
    data=np.argmax(prediction,axis=-1)
    return nib.Nifti1Image(data, affine,hdr)

def run_validation_case(data,affine,hdr,output_dir, model, output_label_map, threshold=0.5, labels=None, overlap=[32,32,16]):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is
    considered a positive result and will be assigned a label.
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    test_data = data
    prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap)

    label_map = output_label_map
    prediction_image = prediction_to_image(prediction, affine, hdr,label_map, threshold=threshold,
                                           labels=labels)
    '''if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        print('haha')
        prediction_image.to_filename(output_dir)'''
    print('haha')
    prediction_image.to_filename(output_dir)

if __name__ == "__main__":
    base_output_dir = '/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_all/pred'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    model_file = '/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_all/final_model.hdf5'
    model = load_old_model(model_file)
    base_img_name = '/media/laomaotao/0A9AD66165F33762/ty/data/test_nii'
    img_folder = os.listdir(base_img_name+'/image')
    label_folder = os.listdir(base_img_name + '/mask')
    for i in range(0,len(img_folder)):
        img_file = os.path.join(base_img_name, 'image', img_folder[i])
        label_file = os.path.join(base_img_name, 'mask', label_folder[i])

        img = nib.load(img_file)
        arr_img = img.get_fdata()
        hdr = nib.Nifti1Header()
        re_affine = img.affine

        arr_img = np.expand_dims(np.asarray(arr_img),axis=-1)
        print(img_folder[i],arr_img.shape)  # mark
        overlap = (32,32,16)

        pred_name = img_folder[i].split('.')[0]+'_pred.nii'
        base_output_file = os.path.join(base_output_dir, pred_name)
        output_label_map = True  # can not determine the use of the parameter
        run_validation_case(arr_img, re_affine, hdr, base_output_file, model, output_label_map,
                                threshold=0.5, labels=(0,1), overlap=overlap)
        # run_validation_case(arr_img, re_affine, hdr, base_output_file, model, output_label_map,
        #                     threshold=0.5, labels=None, overlap=overlap)