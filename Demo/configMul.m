function cnnConfig = configMul()
cnnConfig.layer{1}.type = 'input';
cnnConfig.layer{1}.dimension = [28 28 1];

cnnConfig.layer{2}.type = 'conv';
cnnConfig.layer{2}.filterDim = [5 5];
cnnConfig.layer{2}.numFilters = 10;
cnnConfig.layer{2}.nonLinearType = 'relu';
cnnConfig.layer{2}.conMatrix = ones(1,10);

cnnConfig.layer{3}.type = 'pool';
cnnConfig.layer{3}.poolDim = [2 2];
cnnConfig.layer{3}.poolType = 'maxpool';

cnnConfig.layer{4}.type = 'conv';
cnnConfig.layer{4}.filterDim = [3 3];
cnnConfig.layer{4}.numFilters = 20;
cnnConfig.layer{4}.nonLinearType = 'relu';
cnnConfig.layer{4}.conMatrix = ones(10,20);

cnnConfig.layer{5}.type = 'pool';
cnnConfig.layer{5}.poolDim = [2 2];
%cnnConfig.layer{3}.poolType = 'maxpool';
cnnConfig.layer{5}.poolType = 'maxpool';

cnnConfig.layer{6}.type = 'stack2line';

cnnConfig.layer{7}.type = 'relu';
cnnConfig.layer{7}.dimension = 1280;

cnnConfig.layer{7}.type = 'relu';
cnnConfig.layer{7}.dimension = 640;

cnnConfig.layer{8}.type = 'softmax';
cnnConfig.layer{8}.dimension = 10;

cnnConfig.costFun = 'crossEntropy';
end