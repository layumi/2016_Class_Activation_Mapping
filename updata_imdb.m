load('imdb_voc.mat');
data = imdb.images.data;
for i=1:numel(data)
    tmp = data{i};
    str = strcat('/home/zzd/voc/imgs',tmp(36:end));
    %imshow(imread(str));
    imdb.images.data{i} = str;
end
save('imdb_voc.mat','imdb');