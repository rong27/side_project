function r2 = f_r2(y_train, y_pred);
numerator = sum((y_train-y_pred).^2);
denominator = sum((y_train-mean(y_train)).^2);
r2 = 1 - (numerator/denominator);