clc; % clear command window
clear; % clear Workspace
data_L45 = readtable(".\data_features\L45indexes_XG_avgLen.xlsx");
num_data_L45 = table2array(data_L45);

% randperm 隨機打亂一個數字序列
% s = rng;
n = randperm(size(num_data_L45, 1));

% n = [41, 25, 18, 10, 27, 42, 14, 1, 9, 45, 31, 5, 12, 11, 4, 6, 37, 2, 36, 7, 19, 28, 43, 17, 32, 34, 29, 21, 23, 39, 20, 30, 35, 8, 15, 26, 13, 40, 44, 16, 38, 33, 22, 3, 24]; % DT
% n = [38, 27, 34, 30, 42, 11, 8, 40, 18, 24, 9, 41, 43, 28, 4, 12, 21, 36, 22, 15, 39, 23, 10, 26, 7, 45, 1, 37, 6, 5, 13, 19, 31, 44, 16, 33, 3, 32, 17, 20, 14, 29, 25, 35, 2]; % RF
n = [9, 17, 28, 38, 32, 2, 23, 45, 14, 15, 10, 13, 21, 20, 25, 5, 42, 4, 26, 19, 18, 6, 12, 44, 1, 16, 43, 8, 36, 27, 40, 33, 30, 11, 34, 22, 39, 29, 24, 37, 35, 3, 7, 41, 31]; % XG

% rng(s);

[rows, cols] = size(data_L45);
training_set = 0.7;
checking_set = 0.2;
testing_set = 0.1;
training_num = ceil(rows*training_set);
testing_num = ceil(rows*testing_set);
validaion_num = rows-training_num-testing_num;

% 特徵index
feaures_index_DT = [2:6,8];

training_data = num_data_L45(n(1:training_num), feaures_index_DT);
checking_data = num_data_L45(n(training_num+1:training_num+validaion_num), feaures_index_DT);
testing_data = num_data_L45(n(training_num+validaion_num+1:end), feaures_index_DT);

[rows_f, cols_f] = size(feaures_index_DT);

genOpt = genfisOptions("GridPartition");
genOpt.NumMembershipFunctions = 2;
% genOpt.InputMembershipFunctionType = ["gaussmf", "gbellmf", "gbellmf", "gbellmf", "gbellmf", "gbellmf"];
genOpt.InputMembershipFunctionType = "gaussmf";
genOpt.OutputMembershipFunctionType = "constant";
inFIS = genfis(training_data(:, [1:cols_f-1]), training_data(:, cols_f), genOpt);

epoch = 1000 ;
errorGoal = 0;
opt = anfisOptions("InitialFIS", inFIS, "EpochNumber", epoch, "ErrorGoal", errorGoal);
% opt = anfisOptions("InitialFIS", inFIS, "EpochNumber", epoch, "InitialStepSize", 0.01);

% checking data
opt.ValidationData = checking_data;
training_start = datetime(now,"ConvertFrom","datenum");
[fis2, trainError, stepSize, chkFIS, chkError] = anfis(training_data, opt);
training_end = datetime(now,"ConvertFrom","datenum");

% disp(training_start)
% disp(training_end)

disp("traing time:")
disp(training_end-training_start)

% show rules
% showrule(fis2)
% figure
% plotmf(fis2, "input", 1) % show membership function
% figure
% plotmf(fis2, "input", 2)

% anfis output
predict_training_data = evalfis(fis2, training_data(:, [1:cols_f-1]));
predict_checking_data = evalfis(fis2, checking_data(:, [1:cols_f-1]));
predict_testing_data = evalfis(fis2, testing_data(:, [1:cols_f-1]));

% training data
x = training_data(:, [1:cols_f-1]);
[numRows_train, numCols_train] = size(training_data);
x_axis = (1:numRows_train);
figure
plot(x_axis,training_data(:,cols_f),"or",x_axis,predict_training_data,"*b")
legend("Training Data","ANFIS Output")
title("ANFIS average length prediction (training data)")
xlabel("No") 
ylabel("Length (mm)") 

% checking data
figure
[numRows_check, numCols_check] = size(checking_data);
x_check = (1:numRows_check);
plot(x_check,checking_data(:,cols_f),"or",x_check, predict_checking_data,"*b")
legend("Checking Data","ANFIS Output")
title("ANFIS average length prediction (checking data)")
xlabel("No") 
ylabel("Length (mm)") 

% testing data
figure
[numRows_test, numCols_test] = size(testing_data);
x_test = (1:numRows_test);
plot(x_test,testing_data(:,cols_f),"or",x_test,predict_testing_data,"*b")
legend("Testing Data","ANFIS Output")
title("ANFIS average length prediction (testing data)")
xlabel("No") 
ylabel("Length (mm)") 


% training error
figure
plot((1:epoch), trainError, 'or')
legend("trainError")
title("Training Error")
xlabel("Epoch") 
ylabel("Error (mm)") 

figure
plot((1:epoch), chkError, 'b*')
legend("chkError")
title("Check Error")
xlabel("Epoch") 
ylabel("Error (mm)") 


% RMSE 
% train
y_train = training_data(:, cols_f);
y_hat_train = predict_training_data;
% test
y_test = testing_data(:, cols_f);
y_hat_test = predict_testing_data;
% check
y_check = checking_data(:, cols_f);
y_hat_check = predict_checking_data;

RMSE_train = f_rmse(y_train, y_hat_train)
RMSE_check = f_rmse(y_check, y_hat_check)
RMSE_test = f_rmse(y_test, y_hat_test)
