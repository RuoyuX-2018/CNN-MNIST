function [pooledFeatures, weights] = cnnPool(poolDim, convolvedFeatures, pooltypes)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region, is a 1 * 2 vector(poolDimRow poolDimCol);
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%  weights        - how much the input contributes to the output
%     

if nargin < 3
    pooltypes = 'meanpool';
end

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDimRow = size(convolvedFeatures, 1);
convolvedDimCol = size(convolvedFeatures, 2);
pooledDimRow = floor(convolvedDimRow / poolDim(1));
pooledDimCol = floor(convolvedDimCol / poolDim(2));

weights = zeros(size(convolvedFeatures));
featuresTrim = convolvedFeatures(1:pooledDimRow*poolDim(1),1:pooledDimCol*poolDim(2),:,:);

if strcmp(pooltypes, 'meanpool')
    weights(1:pooledDimRow*poolDim(1), 1:pooledDimCol*poolDim(2),:,:) = ones(size(featuresTrim)) / poolDim(1) / poolDim(2);
end

pooledFeatures = zeros(pooledDimRow, pooledDimCol, numFilters, numImages);

poolFilter = ones(poolDim) * 1/poolDim(1)/poolDim(2);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        features = featuresTrim(:,:,filterNum, imageNum);
        end1 = 1 + poolDim(1) * (pooledDimRow - 1);
        end2 = 1 + poolDim(2)*(pooledDimCol - 1);
        switch pooltypes
            case 'meanpool'
            %To be filled
                fullSize = conv2(features,poolFilter,'valid');
                pooledFeatures(:,:,filterNum,imageNum) = fullSize(1:poolDim(1):end1,1:poolDim(2):end2);
                %pooledFeatures(:,:,filterNum,imageNum) = fullSize(1:poolDim:end,1:poolDim:end);
            case 'maxpool'
                sumRow = 0;
                for i = 1:poolDim(1):end1
                    sumRow = sumRow + 1;
                    sumCol = 0;
                    for j = 1:poolDim(2):end2
                        sumCol = sumCol + 1;
                        a = features(i:(i+poolDim(1)-1),j:(j+poolDim(2)-1));
                        pooledFeatures(sumRow,sumCol,filterNum,imageNum) = maxblock(a);
                        idx_a = find(a == maxblock(a));
                        [jf,ff] = ind2sub([poolDim(1),poolDim(2)],idx_a);
                        weights(i+jf-1,j+ff-1,filterNum,imageNum) = 1;
                    end     
                end
            %To be filled      
            
            otherwise
                error('wrongLayertype: %s',pooltypes);
        end
    end
end


end
 
function b= maxblock(a)
    b = max(max(a));
end

