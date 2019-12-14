%% Convolution Neural Network Exercise

% diary on;
% diary('run_log');

%% STEP 0: Initialize Parameters and Load Data
%  complete the config.m to config the network structure;
cnnConfig = configMul();
%  calling cnnInitParams() to initialize parameters
[theta meta] = cnnInitParams(cnnConfig);

% Load MNIST Data
images = load('train_image');
images = images.train_image;
d = cnnConfig.layer{1}.dimension;
images = reshape(images,d(1),d(2),d(3),[]);
labels = load('train_label');
labels = labels.train_label;
labels(labels==0) = 10; % Remap 0 to 10

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 128;
options.alpha = 1e-1;
options.momentum = .9;

opttheta_mean_single_sigR = minFuncSGD(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta),theta,images,labels,options);
testImages = test_image;
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = test_label;
testLabels(testLabels==0) = 10; % Remap 0 to 10
[cost_mean_single_sigR,grad_mean_single_sigR,preds_mean_single_sigR]=cnnCost(opttheta_mean_single_sigR,testImages,testLabels,cnnConfig,meta,true);
acc_mean_single_sigR = sum(preds_mean_single_sigR==testLabels)/length(preds_mean_single_sigR);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc_mean_single_sigR);