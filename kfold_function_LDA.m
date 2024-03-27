% The kfold_function function calculates the accuracy of k-fold cross-validation 
% for Linear Discriminant Analysis (LDA) classifiers.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date: 19.08.2023

% Inputs:
%   - Data: Matrix of features, where each row represents a sample and each column represents a feature.
%   - Labels: True labels. The length of Labels must be equal to the number of samples in Data.
%   - k: k value in k-fold cross-validation.
%   - step and jj: These two variables are used for representation purposes and are not crucial.
% Outputs:
%   - m_LDA: Mean accuracy of LDA.
%   - std_LDA: Standard deviation of LDA accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kfold_function Function, Coded by Ehsan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% example
% clc;clear;close all; path = '..\';
% load([path,'\MI_IV_2a\CHANN3\dataset_2a_subject1.mat'])
% Data=t_DATA_feat(1:57,1:10);Labels=t_LAB_feat(1:57,1);
% k=5;step=1;jj=1;
% [m_LDA, std_LDA] = kfold_function(Data, Labels, k, step, jj)


function [ m_LDA, std_LDA ] = kfold_function( Data, Labels, k, jj, step )
    for i = 1:k
        % Extract data for the current fold
        Dataa = Data(floor(((i-1)/k) * size(Data,1)) + 1 : floor((i/k) * size(Data,1)), :);
        t_Data = Data;
        t_Data(floor(((i-1)/k) * size(Data,1)) + 1 : floor((i/k) * size(Data,1)), :) = [];
        
        Labelss = Labels(floor(((i-1)/k) * size(Labels,1)) + 1 : floor((i/k) * size(Labels,1)), 1);
        t_Labels = Labels;
        t_Labels(floor(((i-1)/k) * size(Labels,1)) + 1 : floor((i/k) * size(Labels,1)), :) = [];
        
        % Linear Discriminant Analysis (LDA) Classifier
        class = classify(Dataa, t_Data, t_Labels, 'linear');
        d0(i) = 100 * sum(class == Labelss) / length(class);
        
        clc;
        fprintf('\n kfold_function: step is %d index is %d and %d', step, jj, i);
    end
    
    % Calculate Mean and Standard Deviation of the Accuracies
    m_LDA = mean(d0);
    std_LDA = std(d0);
end
