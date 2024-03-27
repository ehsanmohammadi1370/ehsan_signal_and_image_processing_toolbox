% The CSP_ehsan function claculates spatial filter coefficients in common
% spatial patterns algorithm in a tow class classification problem.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% date , 24.07.2023:
% Inputs: Rx : covariance_matrices of class 1 trials.
% Inputs: Ry : covariance_matrices of class 2 trials.
% Outputs: result: spatial filter coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example:
%         Rx = ((X*X')/trace(X*X'));
%         Ry = ((Y*Y')/trace(Y*Y'));
% in BCI problems Rx and Ry are calculated by averaging on all trials
% Rx=is a matrix with size number of EEG channels * number of EEG channels;
% Ry=is a matrix with size number of EEG channels * number of EEG channels;
% [result] = CSP_ehsan(Rx,Ry);
% result is a matrix with the number of EEG channels row and coloumn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   CSP Function,used by ehsan,Coded by James Ethridge and William Weaver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%example 2:write this code in command window then see the result variable
% clc;clear;close all;
% load('..\bci_international_competition_IV_2a\data_making\nA01.mat');
% X=data{1}';Y=data{2}';clear('data');
% Rx = ((X*X')/trace(X*X'));
% Ry = ((Y*Y')/trace(Y*Y'));
% [result] = CSP_ehsan(Rx,Ry);

function [result] = CSP_ehsan(Rx,Ry)
% [result] = CSP_ehsan(Rx,Ry)
if (nargin ~= 2)
        disp('Must have 2 classes for CSP!')
    end
    % Ramoser equation (2)
        Rsum=Ry+Rx;
    
    %   Find Eigenvalues and Eigenvectors of RC
    %   Sort eigenvalues in descending order
    [EVecsum,EValsum] = eig(Rsum);
    aaa=diag(EValsum);
    [EValsum,ind] = sort(aaa,'descend');
    EVecsum = EVecsum(:,ind);
    
    %   Find Whitening Transformation Matrix - Ramoser Equation (3)
        W = sqrt(inv(diag(EValsum))) * EVecsum';  
    
    S{1} = W * Rx * W';%       Whiten Data Using Whiting Transform - Ramoser Equation (4)
    S{2} = W * Ry * W';
    
    % Ramoser equation (5)
   % [U{1},Psi{1}] = eig(S{1});
   % [U{2},Psi{2}] = eig(S{2});
    %generalized eigenvectors/values
    [B,D] = eig(S{1},S{2});
    % Simultanous diagonalization
    % Should be equivalent to [B,D]=eig(S{1});    
     
    %sort ascending by default
    %[Psi{1},ind] = sort(diag(Psi{1})); U{1} = U{1}(:,ind);
    %[Psi{2},ind] = sort(diag(Psi{2})); U{2} = U{2}(:,ind);
    [D,ind]=sort(diag(D));B=B(:,ind);    
    %Resulting Projection Matrix-these are the spatial filter coefficients
    result = B'*W;
end
