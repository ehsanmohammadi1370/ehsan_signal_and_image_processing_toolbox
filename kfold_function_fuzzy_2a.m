% The kfold_function_fuzzy_2a function calculates the accuracy of k-fold cross-validation 
% for fuzzy classifiers: genfis3 and anfis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date: 18.08.2023
% Inputs:
%   - Data: Matrix of features, where each row represents a sample and each column represents a feature.
%   - Labels: True labels. The length of Labels must be equal to the number of samples in Data.
%   - k: k value in k-fold cross-validation.
%   - step and jj: These two variables are used for representation purposes and are not crucial.
%   - method: the input variable of genfis3. See the help for genfis3.
% Outputs:
%   - m_genfis3: Mean accuracy of genfis3.
%   - std_genfis3: Standard deviation of genfis3 accuracy.
%   - m_anfis: Mean accuracy of anfis.
%   - std_anfis: Standard deviation of anfis accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kfold_function_fuzzy_2a Function, Coded by Ehsan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% example
% clc;clear;close all; path = '..\';
% load([path,'\MI_IV_2a\CHANN3\dataset_2a_subject1.mat'])
% Data=t_DATA_feat(1:57,1:10);Labels=t_LAB_feat(1:57,1);
% k=5;step=1;jj=1;method='mamdani';
% [m_genfis3, std_genfis3, m_anfis, std_anfis] = kfold_function_fuzzy_2a(Data, Labels, method, k, step, jj)


function [ m_genfis3, std_genfis3, m_anfis, std_anfis ] = kfold_function_fuzzy_2a( Data, Labels, method, k, jj, step )
    label1 = min(Labels);
    label2 = max(Labels);
    thresh = mean([label1, label2]);
    
    for i = 1:k
        % Extract data for the current fold
        Dataa = Data(floor(((i-1)/k) * size(Data,1)) + 1 : floor((i/k) * size(Data,1)), :);
        t_Data = Data;
        t_Data(floor(((i-1)/k) * size(Data,1)) + 1 : floor((i/k) * size(Data,1)), :) = [];
        
        Labelss = Labels(floor(((i-1)/k) * size(Labels,1)) + 1 : floor((i/k) * size(Labels,1)), 1);
        t_Labels = Labels;
        t_Labels(floor(((i-1)/k) * size(Labels,1)) + 1 : floor((i/k) * size(Labels,1)), :) = [];
        
        % Fuzzy Classifier Training (genfis3)
        fismat = genfis3(t_Data, t_Labels, method);
        output = evalfis(fismat, Dataa);
        ind1 = output > thresh;
        ind2 = output <= thresh;
        output(ind1) = label2;
        output(ind2) = label1;
        acc(i) = sum(Labelss == output) / length(output);
        
        % Fuzzy Classifier Training (ANFIS)
        opt = anfisOptions('InitialFIS', fismat);
        fis = anfis([t_Data t_Labels], opt);
        output2 = evalfis(fis, Dataa);
        ind12 = output2 > thresh;
        ind22 = output2 <= thresh;
        output2(ind12) = label2;
        output2(ind22) = label1;
        acc2(i) = sum(Labelss == output2) / length(output2);
        
        clc;
        fprintf('\n kfold_function: step is %d index is %d and %d', step, jj, i);
    end
    
    % Calculate Mean and Standard Deviation of the Accuracies
    m_genfis3 = mean(acc);
    std_genfis3 = std(acc);
    m_anfis = mean(acc2);
    std_anfis = std(acc2);
end
