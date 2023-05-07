%% index
clc; % clear command window
clear; % clear Workspace
data_L45 = readtable(".\data_features\L45indexes_RF_avgWid.xlsx");
num_data_L45 = table2array(data_L45);

% randperm 隨機打亂一個數字序列
% s = rng;
n = randperm(size(num_data_L45, 1));

% n = [32, 9, 39, 19, 15, 22, 38, 5, 45, 13, 10, 25, 26, 30, 36, 3, 18, 29, 40, 41, 35, 24, 28, 8, 44, 2, 33, 16, 31, 1, 20, 21, 14, 27, 43, 12, 42, 7, 17, 37, 6, 23, 34, 4, 11]; % DT
n = [44, 43, 1, 3, 30, 39, 13, 9, 16, 7, 36, 38, 4, 12, 5, 25, 17, 6, 29, 18, 41, 34, 45, 32, 23, 40, 11, 37, 22, 19, 33, 10, 42, 27, 15, 24, 20, 26, 31, 28, 2, 8, 14, 21, 35]; % RF
% n = [13, 34, 18, 8, 22, 10, 41, 21, 45, 29, 15, 24, 1, 39, 44, 14, 2, 5, 37, 6, 43, 17, 11, 3, 36, 42, 9, 16, 30, 38, 25, 12, 33, 40, 27, 32, 19, 31, 35, 26, 7, 20, 28, 4, 23]; % XG

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
title("ANFIS average width prediction (training data)")
xlabel("No") 
ylabel("Width (mm)") 

% checking data
figure
[numRows_check, numCols_check] = size(checking_data);
x_check = (1:numRows_check);
plot(x_check,checking_data(:,cols_f),"or",x_check, predict_checking_data,"*b")
legend("Checking Data","ANFIS Output")
title("ANFIS average width prediction (ckecking data)")
xlabel("No") 
ylabel("Width (mm)") 

% testing data
figure
[numRows_test, numCols_test] = size(testing_data);
x_test = (1:numRows_test);
plot(x_test,testing_data(:,cols_f),"or",x_test,predict_testing_data,"*b")
legend("Testing Data","ANFIS Output")
title("ANFIS average width prediction (testing data)")
xlabel("No") 
ylabel("Width (mm)") 


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


%%

s = 32
for i = 1:s
    ans = fis2.Outputs.MembershipFunctions(1,i).Parameters;
    ansR = round(ans, 3);
    numAns = sprintf('%d : %g', i, ansR);
    display(numAns)
end 