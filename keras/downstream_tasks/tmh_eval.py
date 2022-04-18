import os
import logging
from utils import segmentation_model_evaluation, plot_image_truth_prediction, \
    get_files, get_samples, load_data
from unet3d import unet_model_3d
from config import ncs_config
import numpy as np


    

def tmh_eval():
    # config
    class set_args():
        gpu = 0
        data = None
        apps = 'ncs'
        run = 1
        cv = None
        subsetting = None
        suffix = 'genesis'
        task = 'segmentation'
        
    args = set_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
    conf = ncs_config(args)

    # dataset

    key = '32x64x64-10'
    input_roots = [
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Image'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Image'),
                ]
    target_roots = [
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Mask'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Mask'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Mask'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Mask'),
                    ]
    
    file_keys = [f'1m{i:04d}' for i in range(37, 45)]
    test_input_samples = get_samples(input_roots, file_keys)   
    test_target_samples = get_samples(target_roots, file_keys) 
    x_test, y_test = load_data(test_input_samples, test_target_samples)
    x_test = x_test[:,np.newaxis]
    y_test = y_test[:,np.newaxis]


    # model
    model = unet_model_3d((1,conf.input_rows,conf.input_cols,conf.input_deps), batch_normalization=True)
    print("[INFO] Load trained model from {}".format( os.path.join(conf.model_path, conf.exp_name+".h5") ))
    # TODO:
    print(os.getcwd())
    model.load_weights(os.path.join('keras', 'downstream_tasks', conf.model_path, conf.exp_name+".h5") )

    # evaluation
    p_test = segmentation_model_evaluation(model=model, config=conf, x=x_test, y=y_test, note=conf.exp_name)

    p_test = np.squeeze(p_test)
    for i in range(0, x_test.shape[0], 1):
        if np.sum( y_test[i])>0:
            name = os.path.split(test_input_samples[i])[1].split('-')[-1][:-4]
            plot_image_truth_prediction(x_test[i], y_test[i], p_test[i], rows=5, cols=5, name=f'figures/keras/{name}_{i:03d}.png')


if __name__ == '__main__':
    tmh_eval()