function [convolvedFeatures, linTrans] = cnnConvolve(images, W, b, nonlineartype, con_matrix, shape)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  images - large images to convolve with, matrix in the form
%           images(row, col, channel, image number)
%  W, b - W, b 
%         W is of shape (filterDim,filterDim,channel,numFilters)
%         b is of shape (numFilters,1)
%  nonlineartype - the type of non-linear type
%         'sigmoid' : default. use sigmoid function
%         'relu'    : rectified linear function
%         'tanh'
%         'softsign'
%  con_matrix -
%         the connection between input channel and output maps. If the ith
%         input channel has connection with jth output map, then
%         con_matrix(i,j) = 1, otherwise, 0;
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

[filterDimRow,filterDimCol,channel,numFilters] = size(W);

if ~exist('con_matrix','var') || isempty(con_matrix)
    con_matrix = ones(channel, numFilters);
end

if ~exist('nonlineartype','var')
    nonlineartype = 'sigmoid';
end

if ~exist('shape','var')
    shape = 'valid';
end

[imageDimRow, imageDimCol,~, numImages] = size(images);
convDimRow = imageDimRow - filterDimRow + 1;
convDimCol = imageDimCol - filterDimCol + 1;

convolvedFeatures = zeros(convDimRow, convDimCol, numFilters, numImages);

%   Convolve every filter with every image here to produce the convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%   To be filled forward pass
for i = 1:numImages
    for j = 1:numFilters
        for k = 1:channel
            if con_matrix(k,numFilters)~=0
                img = images(:,:,k,i);
                filt = W(:,:,k,j);
                filt = rot90(filt,2);
                convolvedFeatures(:,:,j,i) = conv2(img,filt,'valid') + convolvedFeatures(:,:,j,i);
            end
        end
        convolvedFeatures(:,:,j,i) = convolvedFeatures(:,:,j,i) + b(j);
    end
end


linTrans = convolvedFeatures;
switch nonlineartype
    case 'sigmoid'
        convolvedFeatures = 1./(1+exp(-convolvedFeatures));
    case 'relu'
        convolvedFeatures = max(0,convolvedFeatures);
    case 'tanh'
        convolvedFeatures = tanh(convolvedFeatures);
    case 'softsign'
        
        convolvedFeatures = convolvedFeatures ./ (1 + abs(convolvedFeatures));
    case 'none'
        % don't do nonlinearty
    otherwise
        fprintf('error: no such nonlieartype%s',nonlineartype);
end
end

