# Code Description for Face Anti-spoofing code
## 1. Library Requirement
* Prerequisites: Python 3.6, pytorch 1.0.0 or up, Numpy, TensorboardX, Pillow, SciPy, h5py
* Source code files: 
The folder config contains configuration files that describes training setups for all four domain generalization setups in our paper. 
The folder src contains all the python files for running experiments.
The python files starting with "hyper_tune" are training scripts
## 2. Dataset Description and download links
There are four datasets used in our experiment, CASIA, Replay-attack, Oulu and MSU dataset. The datasets are introduced in following papers:   
### CASIA Dataset 
"A face antispoofing database with diverse attacks."  
```
@inproceedings{zhang2012face,
  title={A face antispoofing database with diverse attacks},
  author={Zhang, Zhiwei and Yan, Junjie and Liu, Sifei and Lei, Zhen and Yi, Dong and Li, Stan Z},
  booktitle={2012 5th IAPR international conference on Biometrics (ICB)},
  pages={26--31},
  year={2012},
  organization={IEEE}
}
```
### Oulu-npu Dataset 
"Oulu-npu: A mobile face presentation attack database with real-world variations."      
```
@inproceedings{boulkenafet2017oulu,
  title={Oulu-npu: A mobile face presentation attack database with real-world variations},
  author={Boulkenafet, Zinelabinde and Komulainen, Jukka and Li, Lei and Feng, Xiaoyi and Hadid, Abdenour},
  booktitle={2017 12th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2017)},
  pages={612--618},
  year={2017},
  organization={IEEE}
}

```
### Replay Attack Dataset  
"On the effectiveness of local binary patterns in face anti-spoofing."
```
@inproceedings{chingovska2012effectiveness,
  title={On the effectiveness of local binary patterns in face anti-spoofing},
  author={Chingovska, Ivana and Anjos, Andr{\'e} and Marcel, S{\'e}bastien},
  booktitle={2012 BIOSIG-proceedings of the international conference of biometrics special interest group (BIOSIG)},
  pages={1--7},
  year={2012},
  organization={IEEE}
}

```
### MSU Dataset
"Face spoof detection with image distortion analysis."
```
@article{wen2015face,
  title={Face spoof detection with image distortion analysis},
  author={Wen, Di and Han, Hu and Jain, Anil K},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={10},
  number={4},
  pages={746--761},
  year={2015},
  publisher={IEEE}
}
```

## 3. Training Steps
There's a sample of training command in file src/command_example.py, it shows a typical command structure. 
### Structure of training file
The major python file for the proposed method is src/train_dg_final.py, where both training and testing process are conducted simultaneously along the training process. Most of the possible options are listed in as arguments and have default values. There are some parameters that requires to be specified, and they are listed below:  
  
    
config_path: 
It specified the location of the configuration file, which should consist of the location of training and test dataset, hyper-parameters for optimizer and other network structures.   
out_path: 
The location to store all training informations, including checkpoints and yaml files to store the hyper-parameters for each single training attempt.     
project_path: 
The location for evaluation results, including classification scores for both validation and test dataset, as well as the computed HTER and AUC results.  
  
A typical experiment command should look like this: 
```
python train_dg_final.py --config_path <the location of config file> --out_path <path to info> --project_path <path to result> 
```

### Structure of configuration file
Another important file is the configuration file which contains the dataset configurations, there should be two configuration files, data_loader_dg.yaml and the train.yaml file for training. 

### Training yaml components
In the file train.yaml, there should be a dataset configuration description:  
For example: 
```
test_dataset_conf:
  oulu-npu:
    dataset_path_machine0:           '<Path to your oulu-npu dataset>'
    img_path_machine0:               '<Path to your oulu-npu dataset>/rgb_images/test'
    img_path_dev_machine0:           '<Path to your oulu-npu dataset>/rgb_images/dev'

  replay-attack:
    dataset_path_machine0:  '<Path to your replay-attack dataset>/replay-attack'
    full_img_path_machine0: '<Path to your replay-attack dataset>/rgb_images_full/train'
    full_img_path_dev_machine0: '<Path to your replay-attack dataset>/rgb_images_full/dev'
    full_img_path_test_machine0: '<Path to your replay-attack dataset>/rgb_images_full/test'

  casia:
    dataset_path_machine0: '<Path to your casia dataset>'
    full_img_path_machine0: '<Path to your casia dataset>/rgb_images_full/train'
    full_img_path_test_machine0: '<Path to your casia dataset>/rgb_images_full/test'

  msu:
    dataset_path_machine0: '<Path to your msu dataset>'
    full_img_path_machine0: '<Path to your msu dataset>/rgb_images_full/train'
    full_img_path_test_machine0: '<Path to your msu dataset>/rgb_images_full/test'

```
Also, the configure file data_loader_dg.yaml, data paths should be changed accordingly. Moreover, we use dlib to extract facial bounding boxes for all the dataset, and the path to facial bounding box txt files should be included also. 

All training process are set to have 100k iterations by default. 
## 4. Test Steps
The test phases are conducted whenever an epoch of training is finished. As instructed it is conducted automatically and does not require any further input of parameters. The final HTER can be found under "--project_path" folder. 

