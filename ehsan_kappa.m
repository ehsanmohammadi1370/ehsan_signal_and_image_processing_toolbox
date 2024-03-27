% The ehsan_kappa function claculates cohens's kappa coefficient , which is a statistical
% measure of inter-rater agreement. It is often used to assess the agreement between two
% raters or algorithms in classifying data into multiple classes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% date , 24.07.2023:
% Inputs:
% true_labels : True Labels of trials.
% predicted_labels : Predicted Labels of trials by the algorithm.
% num_cl : number of classes in the algorithm.
% Outputs:
% coh_kappa: Cohen's kappa coefficient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example:
% true_labels = [1,2,1,2,1,2,1,2,1,2]; 
% predicted_labels = [2,2,1,2,1,2,1,2,1,1];
% num_cl = 2;
% coh_kappa = ehsan_kappa(true_labels, predicted_labels, num_cl); 
% coh_kappa = 0.6000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function coh_kappa = ehsan_kappa(true_labels, predicted_labels, num_cl)
    cc = zeros(num_cl, num_cl);  % Initialize the confusion matrix.

    % Calculate the confusion matrix.
    for i = 1:length(true_labels)
        row = true_labels(i);
        column = predicted_labels(i);
        cc(row, column) = cc(row, column) + 1;   
    end

    f = diag(ones(1, num_cl));  % Create a matrix with ones on the diagonal.
    n = sum(cc(:));  % Sum of the matrix elements (total number of trials).
    cc = cc ./ n;  % Convert the matrix to proportions (probabilities).
    r = sum(cc, 2);  % Sum of each row (proportions of true labels).
    s = sum(cc);  % Sum of each column (proportions of predicted labels).
    Ex = r * s;  % Calculate the expected proportion for random agreement.

    pom = sum(min([r'; s]));  % Sum of the minimum values of each row and column.
    po = sum(sum(cc .* f));  % Sum of the diagonals of the confusion matrix (sum of true positives).
    pe = sum(sum(Ex .* f));  % Sum of the expected diagonals for random agreement.

    coh_kappa = (po - pe) / (1 - pe);  % Calculate Cohen's kappa coefficient.
end
