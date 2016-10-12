function train_id_net_res(varargin)
addpath('../MATLAB');
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./SiftFlowDataset/sift_data_256.mat') ;
imdb = imdb.imdb;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_ims_MAC();
%rmsprop

for i=1:numel(net.params)
    if(strcmp(net.params(i).trainMethod ,'gradient'))
        net.params(i).trainMethod = 'rmsprop';
    end
end
%}
net.params(net.getParamIndex('final_filter')).learningRate = 0.01;
net.conserveMemory = true;
net.meta.normalization.averageImage = mean(mean(mean(imdb.images.data,4),1),2);

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 64;
opts.train.continue = false; 
opts.train.gpus = 3;
opts.train.prefetch = false ;
opts.train.gamma = 0.9;
opts.train.constraint = 2;
opts.train.expDir = './data/resnet52_drop0.9' ;
opts.train.learningRate = [1e-3*ones(1,40)] ; %1e-5
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ; 
labels = imdb.images.label(batch,:) ;% 64*33
batchsize = size(im,4);
oim = zeros(224,224,3,batchsize,'single');
for i=1:batchsize
    x = randi(33);
    y = randi(33);
    temp = im(x:x+223,y:y+223,:,i);
    r = rand>0.5;
    if r 
        oim(:,:,:,i) = temp;
    else oim(:,:,:,i) = fliplr(temp);
    end
end
oim = bsxfun(@minus,oim,opts.averageImage); 
labels = reshape(labels',1,1,33,[]);  
inputs = {'data',gpuArray(oim),'label',labels};
