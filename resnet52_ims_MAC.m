function net = resnet52_ims_MAC()
netStruct = load('./data/imagenet-resnet-50-dag.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
for i = 175:-1:141
    net.removeLayer(net.layers(i).name);
end

for i = 1:numel(net.params)
    if(mod(i,2)==0)
        net.params(i).learningRate=0.02;
    else net.params(i).learningRate=0.001;
    end
    net.params(i).weightDecay=1;
end
% average pooling
net.addLayer('MAC',dagnn.MAC(),{'res4fx'},{'res4fxm'},{});  
dropoutBlock = dagnn.DropOut('rate',0.9);
net.addLayer('dropout',dropoutBlock,{'res4fxm'},{'res4fxd'},{});
fc751Block = dagnn.Conv('size',[1 1 1024 20],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc',fc751Block,{'res4fxd'},{'prediction'},{'final_filter'});

lossBlock = dagnn.ODistLoss();
net.addLayer('contrastive_loss',lossBlock,{'prediction','label'},'objective');
net.initParams();
%net.conserveMemory = false;
%net.eval({'data',rand(224,224,3,'single')});
end

