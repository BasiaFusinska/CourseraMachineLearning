function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

dim = 3;

par_val = [0.1, 0.3, 1];

min_error = 0;
min_c = 0;
min_s = 0;

for i = 1:dim
  for j = 1:dim
  
    c = par_val(i);
    s = par_val(j);
    
    model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
    
    predictions = svmPredict(model, Xval);

    e = mean(double(predictions -= yval));
    uns_e = e^2;
            
    if (i == 1 && j == 1) || uns_e < min_error
      min_error = uns_e;
      min_c = c;
      min_s = s;
    end
  end
end

C = min_c;
sigma = min_s;

% =========================================================================

end
