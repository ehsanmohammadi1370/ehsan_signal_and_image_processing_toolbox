% The kfold_function_2classifier function calculates the accuracy of k-fold cross-validation 
% for Random Forest (RF) and K-Nearest Neighbors (KNN).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date: 18.08.2023
% Please note: The path to the Random Forest folder must be added to MATLAB paths.
% pp = genpath('.\randomforest');
% addpath(pp);
% Inputs:
%   - Data: Matrix of features, where each row represents a sample and each column represents a feature.
%   - Labels: True labels. The length of Labels must be equal to the number of samples in Data.
%   - k: k value in k-fold cross-validation.
%   - step and jj: These two variables are used for representation purposes and are not crucial.
% Outputs:
%   - m_RF: Mean accuracy of RF.
%   - std_RF: Standard deviation of RF accuracy.
%   - m_KNN: Mean accuracy of KNN.
%   - std_KNN: Standard deviation of KNN accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kfold_function Function, Coded by Ehsan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% example
% clc;clear;close all; path = '..\';
% pp = genpath([path,'functions\randomforest']);
% addpath(pp);
% load([path,'\MI_IV_2a\CHANN3\dataset_2a_subject1.mat'])
% Data=t_DATA_feat(1:57,1:10);Labels=t_LAB_feat(1:57,1);
% k=5;step=1;jj=1;
% [m_RF, std_RF, m_KNN, std_KNN] = kfold_function_2classifier(Data, Labels, k, step, jj)

function [m_RF, std_RF, m_KNN, std_KNN] = kfold_function(Data, Labels, k, step, jj)
for i = 1:k
    % Splitting the data and labels into training and testing sets for the current fold.
    Dataa = Data(floor(((i - 1) / k) * size(Data, 1)) + 1 : floor((i / k) * size(Data, 1)), :);
    t_Data = Data;
    t_Data(floor(((i - 1) / k) * size(Data, 1)) + 1 : floor((i / k) * size(Data, 1)), :) = [];
    Labelss = Labels(floor(((i - 1) / k) * size(Labels, 1)) + 1 : floor((i / k) * size(Labels, 1)), 1);
    t_Labels = Labels;
    t_Labels(floor(((i - 1) / k) * size(Labels, 1)) + 1 : floor((i / k) * size(Labels, 1)), :) = [];

    % Random Forest classifier
    model = forestTrain(t_Data, t_Labels);
    [Yhard1(floor(((i - 1) / k) * size(Labels, 1)) + 1 : floor((i / k) * size(Labels, 1)), 1), Ysoft] = forestTest(model, Dataa);
    a0(i) = (sum(Yhard1(floor(((i - 1) / k) * size(Labels, 1)) + 1 : floor((i / k) * size(Labels, 1)), 1) == Labelss) / length(Labelss)) * 100;

    % K-Nearest Neighbors classifier
    NumNeighbors = 10;
    Mdl_Knn1 = fitcknn(t_Data, t_Labels, 'NumNeighbors', NumNeighbors, 'Distance', 'euclidean', 'Standardize', 1);
    dicted_label1(floor(((i - 1) / k) * size(Labels, 1)) + 1 : floor((i / k) * size(Labels, 1)), 1) = predict(Mdl_Knn1, Dataa);
    c0(i) = (sum(dicted_label1(floor(((i - 1) / k) * size(Labels, 1)) + 1 : floor((i / k) * size(Labels, 1)), 1) == Labelss) / length(Labelss)) * 100;

    clc;
    fprintf('\n kfold_function: index kfold is %d step1 is %d  and step2 is %d', i, step, jj);
end

% Calculate mean and standard deviation of accuracy for each classifier.
m_RF = mean(a0); std_RF = std(a0);
m_KNN = mean(c0); std_KNN = std(c0);

end
