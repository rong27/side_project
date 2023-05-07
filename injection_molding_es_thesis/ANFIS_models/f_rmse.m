function [rmse, error, error_squared, mse] = f_rmse(y_train, y_pred)
error = y_train-y_pred;
error_squared = error.^2;
mse = mean(error_squared);
rmse = sqrt(mse);


% y_train = training_data(:, cols);
% y_hat_train = predict_training_data;
% 算式
% error_train = y_train-y_hat_train;
% error_squared_train = error_train.^2;
% mse_train = mean(error_squared_train);
% RMSE_train = sqrt(mse_train)