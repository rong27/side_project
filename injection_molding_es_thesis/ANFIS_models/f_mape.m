function mape = f_mape(y_train, y_pred);
mape = mean(abs(y_train-y_pred))*100;