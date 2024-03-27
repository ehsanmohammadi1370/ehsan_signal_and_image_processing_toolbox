% The ehsan_PLV function Compute the Phase Locking Value between two signals across trials, according to Lachaux, 
% Rodriguez, Martinerie, and Varela (1999). The PLV value ranges from 0, indicating random 
% phase differences, to 1 indicating a fixed phase difference. 
% phase_sig1 and phase_sig2 should be the phase values of the signals in radians, arranged as
% Samples x Trials. These can bed computed using the Wavelet or Hilbert transform, for example:
% phase_sig = angle(hilbert(BPS)); 
% Where BPS is the signal after band-pass filtering around the frequency range of interest. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date: 18.08.2023
% Inputs:
%   - sig1: signal number one.
%   - sig2: signal number two.
%   - order: order of the FIR filter.
%   - range: frequency range of interest.
%   - Fs: sampling frequency.
% Outputs:
%   - plv: Phase Locking Value between the two signals.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example:
%   range = [4 8];     % Frequency range of interest.
%   order = floor(1000 / mean(range));   % Order of the FIR filter.
%   Fs = 250;           % Sampling frequency.
%   [plv] = ehsan_PLV(x(i,:), x(j,:), order, range, Fs);
%   x is an EEG matrix, and i and j are two different channels.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REFERENCE: https://www.mathworks.com/matlabcentral/fileexchange/71739-plv-phase-locking-value
% ehsan_PLV Function, used by ehsan, but written by Edden Gerber in 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%example 2:write this code in command window then see the plv variable
% clc;clear;close all;
% load('..\bci_international_competition_IV_2a\data_making\nA01.mat');
% X=data{1}';clear('data');
% sig1=X(1,:);sig2=X(2,:);
% range = [4 8];order = floor(1000 / mean(range));Fs = 250;
% [plv] = ehsan_PLV(sig1, sig2, order, range, Fs)

function [plv] = ehsan_PLV(sig1, sig2, order, range, Fs)
% Design an FIR filter for the specified frequency range.
    filtPts = fir1(order, 2/Fs * range);
    % Apply the filter to the signals.
    BPS1 = filter(filtPts, 1, sig1, [], 2);
    BPS2 = filter(filtPts, 1, sig2, [], 2);
    % Compute the phase values using the Hilbert transform.
    phase_sig1 = angle(hilbert(BPS1));
    phase_sig2 = angle(hilbert(BPS2));
    % Transpose the phase matrices for computation.
    phase_sig1 = phase_sig1';
    phase_sig2 = phase_sig2';
    % Compute the PLV.
    e = exp(1i * (phase_sig1 - phase_sig2));
    plv = abs(sum(e)) / length(sig1);
end