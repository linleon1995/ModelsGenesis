import os
import random
import copy
import keras
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import backend as K
from sklearn import metrics
from keras.callbacks import LambdaCallback,TensorBoard,ReduceLROnPlateau

###
import logging

def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None, recursive=True, get_dirs=False, ignore_suffix=False):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
        recursive: The flag for searching path recursively or not(bool)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_back_filelist(root, f, file_list, is_fullpath):
        f = f[:-4] if ignore_suffix else f
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # print(root, dirs, files)
        if not recursive:
            if i > 0: break

        if get_dirs:
            files = dirs
            
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_back_filelist(root, f, file_list, return_fullpath)
            else:
                push_back_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        f = 'dir' if get_dirs else 'file'
        if keys: 
            logging.warning(f'No {f} exist with key {keys}.') 
        else: 
            logging.warning(f'No {f} exist.') 
    return file_list


def get_samples(roots, cases):
    samples = []
    for root in roots:
        samples.extend(get_files(root, keys=cases))
    return samples


def load_data(input_samples, target_samples, remove_zeros):
    total_input_data, total_target = [], []
    pid_dict = {}
    for idx, (x_path, y_path) in enumerate(zip(input_samples, target_samples)):
        # print(idx)
        input_data = np.load(x_path)
        target = np.load(y_path)

        if remove_zeros:
            if not np.sum(target):
                continue

        # if remove_shift:
        #     file_name = os.path.split(x_path)[1][:-4]
        #     pid = file_name.split('-')[-1]
        #     case_num = int(file_name.split('-')[1])
        #     if pid in pid_dict:
        #         continue
        #     else:
        #         pid_dict[pid] = case_num

        input_data, target = np.swapaxes(np.swapaxes(input_data, 0, 1), 1, 2), np.swapaxes(np.swapaxes(target, 0, 1), 1, 2)
        input_data = input_data / 255
        input_data, target = input_data[np.newaxis], target[np.newaxis]

        total_input_data.append(input_data)
        total_target.append(target)

    total_input_data = np.concatenate(total_input_data, axis=0)
    total_target = np.concatenate(total_target, axis=0)
    return total_input_data, total_target
###

def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2)
    return data_sample

def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample

######
# Module: Evaluation metric
######
def dice_eval(y_true, y_pred):
    y_true = tf.where(tf.greater(y_true, 0.5), tf.ones_like(y_true), tf.zeros_like(y_true))
    y_pred = tf.where(tf.greater(y_pred, 0.5), tf.ones_like(y_pred), tf.zeros_like(y_pred))
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    summation = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    summation = tf.maximum(summation, 1e-10)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # print(y_true.shape, y_pred.shape, summation.shape)
    # print(y_true_f.shape, y_pred_f.shape, intersection.shape)
    iou = 2 * intersection / summation
    return tf.minimum(iou, 1)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def iou(im1, im2):
    overlap = np.uint8(im1>0.5) * np.uint8(im2>0.5)
    union = np.uint8(im1>0.5) + np.uint8(im2>0.5) - overlap
    return overlap.sum()/float(union.sum())
        
def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1>0.5).astype(np.bool)
    im2 = np.asarray(im2>0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum



######
# Module: model setup
######
def classification_model_compile(model, config):
    if config.num_classes <= 2:
        model.compile(optimizer=config.optimizer, 
                      loss="binary_crossentropy", 
                      metrics=['accuracy','binary_crossentropy'],
                 )
    else:
        model.compile(optimizer=config.optimizer, 
                      loss="categorical_crossentropy", 
                      metrics=['categorical_accuracy','categorical_crossentropy'],
                 )
    return model

def segmentation_model_compile(model, config):
    model.compile(optimizer=config.optimizer, 
                  loss=dice_coef_loss, 
                  metrics=[mean_iou, 
                           dice_coef],
                 )
    return model
    
def model_setup(model, config, task=None):
    if task == 'segmentation':
        model = segmentation_model_compile(model, config)
    elif task == 'classification':
        model = classification_model_compile(model, config)
    else:
        raise

    if os.path.exists(os.path.join(config.model_path, config.exp_name+".txt")):
        os.remove(os.path.join(config.model_path, config.exp_name+".txt"))
    with open(os.path.join(config.model_path, config.exp_name+".txt"),'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    shutil.rmtree(os.path.join(config.logs_path, config.exp_name), ignore_errors=True)
    if not os.path.exists(os.path.join(config.logs_path, config.exp_name)):
        os.makedirs(os.path.join(config.logs_path, config.exp_name))
    tbCallBack = TensorBoard(log_dir=os.path.join(config.logs_path, config.exp_name),
                             histogram_freq=0,
                             write_graph=True, 
                             write_images=True,
                            )
    tbCallBack.set_model(model)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=config.patience, 
                                                   verbose=config.verbose,
                                                   mode='min',
                                                  )
    check_point = keras.callbacks.ModelCheckpoint(os.path.join(config.model_path, config.exp_name+".h5"),
                                                  monitor='val_loss', 
                                                  verbose=config.verbose, 
                                                  save_best_only=True, 
                                                  mode='min',
                                                 )
    lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6,
                                        min_delta=0.0001, min_lr=1e-6, verbose=1)
    callbacks = [check_point, early_stopping, tbCallBack, lrate_scheduler]
    return model, callbacks

def classification_model_evaluation(model, config, x, y, note=None):
    model = classification_model_compile(model, config)
    p = model.predict(x, verbose=config.verbose, batch_size=config.batch_size)
    if note is not None:
        print("[INFO] {}".format(note))
    print("x:  {} | {:.1f} ~ {:.1f}".format(x.shape, np.min(x), np.max(x)))
    print("y:  {} | {:.1f} ~ {:.1f}".format(y.shape, np.min(y), np.max(y)))
    print("p:  {} | {:.1f} ~ {:.1f}".format(p.shape, np.min(p), np.max(p)))
    
    fpr, tpr, thresholds = metrics.roc_curve(y, p, pos_label=1)
    
    print("[EVAL] AUC = {:.2f}%".format(100.0 * metrics.auc(fpr, tpr)))


def segmentation_model_evaluation(model, config, x, y, note=None):
    model.compile(optimizer=config.optimizer, 
                  loss=dice_coef_loss, 
                  metrics=[mean_iou, 
                           dice_eval],
                 )
    p = model.predict(x, verbose=config.verbose, batch_size=config.batch_size)

    total_dice = 0
    for p_sample, y_sample in zip(p, y):
        total_dice += dice(p_sample, y_sample)
    total_dice /= p.shape[0]
    print(f'Dice {100*total_dice:.2f} %')

    eva = model.evaluate(x, y, verbose=config.verbose, batch_size=config.batch_size)
    if note is not None:
        print("[INFO] {}".format(note))
    print("x:  {} | {:.1f} ~ {:.1f}".format(x.shape, np.min(x), np.max(x)))
    print("y:  {} | {:.1f} ~ {:.1f}".format(y.shape, np.min(y), np.max(y)))
    print("p:  {} | {:.1f} ~ {:.1f}".format(p.shape, np.min(p), np.max(p)))
    # print("[BIN]  Dice = {:.2f}%".format(100.0 * dice(p, y)))
    # print("[BIN]  IoU  = {:.2f}%".format(100.0 * iou(p, y)))
    print("[EVAL] Dice = {:.2f}%".format(100.0 * eva[-1]))
    print("[EVAL] IoU  = {:.2f}%".format(100.0 * eva[-2]))
    
    return p

######
# Module: Visualization
######

def plot_image_truth_prediction(x, y, p, rows=12, cols=12, name=None):
    x, y, p = np.squeeze(x), np.squeeze(y), np.squeeze(p>0.5)
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(25*3, 25))

    large_image = np.zeros((rows*x.shape[0], cols*x.shape[1]))
    for b in range(rows*cols):
        large_image[(b//rows)*x.shape[0]:(b//rows+1)*x.shape[0],
                    (b%cols)*x.shape[1]:(b%cols+1)*x.shape[1]] = np.transpose(np.squeeze(x[:, :, b]))
    plt.subplot(1, 3, 1)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1); plt.axis('off')

    large_image = np.zeros((rows*x.shape[0], cols*x.shape[1]))
    for b in range(rows*cols):
        large_image[(b//rows)*y.shape[0]:(b//rows+1)*y.shape[0],
                    (b%cols)*y.shape[1]:(b%cols+1)*y.shape[1]] = np.transpose(np.squeeze(y[:, :, b]))
    plt.subplot(1, 3, 2)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1); plt.axis('off')

    large_image = np.zeros((rows*p.shape[0], cols*p.shape[1]))
    for b in range(rows*cols):
        large_image[(b//rows)*p.shape[0]:(b//rows+1)*p.shape[0],
                    (b%cols)*p.shape[1]:(b%cols+1)*p.shape[1]] = np.transpose(np.squeeze(p[:, :, b]))
    plt.subplot(1, 3, 3)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1); plt.axis('off')

    if name is not None:
        plt.savefig(name)
    else:
        plt.show()      
    
    
def plot_case(case_id=None, mris=None, segs=None, rows=10, cols=10, increment=38):
    assert case_id is not None
    assert mris is not None
    assert segs is not None
    font = {'family' : 'times',
            'weight' : 'bold',
            'size'   : 22}
    plt.rc('font', **font)
    
    print("\n\n[INFO] case id {}".format(case_id))
    
    # plot the patient MRI
    plt.figure(figsize=(cols*1, rows*1))
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.transpose(mris[case_id, 0, :, :, i+increment]), cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
    plt.show()

    # plot the segmentation mask
    plt.figure(figsize=(cols*1, rows*1))
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.transpose(segs[case_id, 0, :, :, i+increment]), cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
