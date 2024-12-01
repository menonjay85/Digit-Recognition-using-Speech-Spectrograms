clc;
clear all;
close all;

No_train = 200;     % No. of training points
No_test = 50;       %No. of test points
hiddenlayersize = [6 6];


[X_train,y_train] = generate_dataset(No_train);
gscatter(X_train(:,1), X_train(:,2), y_train);
title('XOR Train Dataset');
figure;

[X_test,y_test] = generate_dataset(No_test);
gscatter(X_test(:,1), X_test(:,2), y_test);
title('XOR Test Dataset');
figure;

net = neural_network(hiddenlayersize, X_train, y_train);

confusionmatrix(net, X_train, y_train, 'Training');
figure;
confusionmatrix(net, X_test, y_test, 'Testing');


function confusionmatrix(net,X,y,name)
    
    %Neural network output
    outputs = net(X');
    
    %Training Dataset Confusion Matrix
    plotconfusion(y',outputs);
    title(sprintf('XOR %s Dataset Confusion Matrix', name));
end
    

function net = neural_network(hiddenlayersize, X, y)
    %% Neural network training
    trainFcn = 'trainscg';

    % Hidden Layer dimensions (Can be customized)
    hiddenLayerSize = hiddenlayersize;

    %Neural network object
    net = patternnet(hiddenLayerSize, trainFcn);
    net.trainParam.showWindow = 0;  

    %Training parameters
    %Here we are training on the entire dataset
    net.divideParam.trainRatio = 1;

    %Training process
    [net,tr] = train(net,X',y');
end

function [xor_dataset,labels] = generate_dataset(N)
    
    %N no. of points generated from a normal distribution
    xor_dataset = randn(N,2);  
    
    %Performing XOR operation on the columns and converting logical variables to double data type
    y_xor = double(xor((xor_dataset(:,1)>0), (xor_dataset(:,2)>0)));
    
    %Labels of the dataset
    labels = y_xor;  
    
end
    


    

