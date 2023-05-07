function mae = f_mae(y_train, y_pred);
mae = mean(abs(y_train-y_pred));