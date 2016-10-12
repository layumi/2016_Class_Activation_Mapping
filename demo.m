clear;

im = imread('./coast_arnat59.jpg');  % coast_arnat59.jpg
tags = importdata('./labelList_voc.txt');
heatmap = [];
subtitle = [];

netStruct = load('./data/resnet52_drop0.9/net-epoch-40.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
net.mode = 'test' ;
net.conserveMemory = false;
net.move('gpu') ;
im_mean = net.meta.normalization.averageImage;
im = imresize(im,[224,224]);
im_data = bsxfun(@minus,single(im),im_mean);
net.eval({'data',gpuArray(im_data)});
input_hp = net.vars(net.getVarIndex('res4fx')).value;
weight = net.params(net.getParamIndex('final_filter')).value;
score = net.vars(net.getVarIndex('prediction')).value;

input_hp = reshape(gather(input_hp),[],1024);  %16*16*1024 ->  216*1024
weight = reshape(gather(weight),1024,20);
[s,index] = sort(score(:),'descend');

for i=1:8
    j = index(i);
    hp = input_hp * weight(:,j);  %216*1
    hp = reshape(hp,14,14);
    max_value = max(hp(:));
    mapIm = mat2im(hp, jet(100), [0 max_value]);
    imToShow = mapIm*0.5 + (single(imresize(im,[14,14]))/255)*0.5;
    %imshow(imToShow);
    heatmap = cat(4,heatmap,imToShow);
    subtitle = cat(1,subtitle,tags(j));
end

subplot(3,3,1);
imshow(im);
for i = 1:8
    subplot(3,3,i+1);
    imshow(heatmap(:,:,:,i));
    title(sprintf('%s:%.3f',subtitle{i},s(i)));
end


%{
a = ones(1,33)./33;
b = zeros(1,33); b(1) = 1;
r = (a-b).^2 *64;
sum(r)
%}