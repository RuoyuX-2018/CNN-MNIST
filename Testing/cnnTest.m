%% STEP 1: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = test_image;
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = test_label;
testLabels(testLabels==0) = 10; % Remap 0 to 10
cnnConfig = config();
[theta meta] = cnnInitParams(cnnConfig);
[cost,grad,preds]=cnnCost(theta,testImages,testLabels,cnnConfig,meta,true);

acc = sum(preds==testLabels);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

%diary off;