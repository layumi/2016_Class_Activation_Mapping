# Class Activation Mapping (Matlab) 
Idea is from Learning Deep Features for Discriminative Localization(CVPR16)

code: https://github.com/metalbubble/CAM

I realize it in MATLAB and use resnet50 as pretrained model.
VOC2012 Dataset is used to train the model. (SiftFlow Dataset is considered to be too small to train.)

#To test
1.run `gpu_compile.m`  to compile matconvnet.

2.run `demo.m`

#To train
1.run `gpu_compile.m`  to compile matconvnet.

2.run `train_id_net_res_voc.m`  (you should change image path in the 'imdb_voc.mat')
