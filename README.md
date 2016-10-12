# Class Activation Mapping (Matlab) 
Idea is from Learning Deep Features for Discriminative Localization(CVPR16)

I realize it in MATLAB and use resnet50 as pretrained model.
![](https://github.com/layumi/2016_Class_Activation_Mapping/blob/master/show.jpeg)

#Data
VOC2012 Dataset is used to train the model.
(SiftFlow Dataset is considered to be too small to train.)

#To test
1.compile matconvnet. You may just uncomment some lines in `gpu_compile.m`.

2.run `demo.m` to have fun.

#To train
1.run `gpu_compile.m`  to compile matconvnet.

2.run `train_id_net_res_voc.m`  (You should change image path in the 'imdb_voc.mat')
