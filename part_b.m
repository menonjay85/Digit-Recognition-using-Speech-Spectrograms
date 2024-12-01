load noisydigitrecognition.mat; 
load labels.mat;

rand_ind = randperm(10000);
data = noisydigitrecognition(rand_ind, :); 
labels = labels(rand_ind, :); 

figure;
num_samples_to_plot = 10; 
for i = 1:num_samples_to_plot
    subplot(2, 5, i);
    imagesc(reshape(data(i, :), [28, 28])); 
    % colormap('gray');
    axis off; 
    title(['Sample ', num2str(i)]);
end
sgtitle('Visualization of Noisy Digit Recognition Data');


% Test train split
num_train = round(0.6 * size(data, 1));
X_train = data(1:num_train, :);
y_train = labels(1:num_train, :);
X_test = data(num_train+1:end, :);
y_test = labels(num_train+1:end, :);

trainFcn = 'trainscg'; 
hiddenLayerSize = [12 16 12]; 
net = patternnet(hiddenLayerSize, trainFcn);

net.trainParam.showWindow = 0; 
net.divideParam.trainRatio = 0.8; 
net.divideParam.valRatio = 0.2; 
net.divideParam.testRatio = 0; 

[net, tr] = train(net, X_train', y_train');

% Testing
y_pred = net(X_test');
y_pred_classes = vec2ind(y_pred); % Prediction
y_test_classes = vec2ind(y_test'); 

accuracy = sum(y_pred_classes == y_test_classes) / length(y_test_classes) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

figure;
plotperform(tr);
title('Training Performance'); % optional

% Confusion matrix
figure;
confMat = confusionchart(y_test_classes, y_pred_classes);
confMat.Title = 'Confusion Matrix';
confMat.RowSummary = 'row-normalized';
confMat.ColumnSummary = 'column-normalized';
