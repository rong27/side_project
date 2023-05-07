clc; % clear command window
clear; % clear Workspace
data_L45 = readtable(".\data_features\L45indexes_DT_weight.xlsx");
num_data_L45 = table2array(data_L45);

% randperm 隨機打亂一個數字序列
% s = rng;
n = randperm(size(num_data_L45, 1));

n = [26, 29, 7, 12, 41, 2, 30, 19, 21, 4, 10, 17, 15, 5, 31, 6, 3, 44, 32, 18, 11, 8, 14, 22, 38, 23, 1, 13, 43, 35, 28, 37, 33, 42, 20, 40, 9, 45, 39, 16, 24, 34, 25, 36, 27]; % DT
% n = [32, 17, 16, 42, 7, 36, 38, 33, 6, 31, 26, 11, 45, 25, 44, 23, 20, 2, 8, 43, 40, 27, 5, 3, 13, 19, 41, 39, 28, 15, 10, 24, 22, 4, 35, 30, 18, 9, 14, 21, 34, 29, 1, 37, 12]; % RF
% n = [43, 10, 7, 41, 38, 34, 27, 31, 9, 19, 21, 37, 20, 22, 39, 6, 13, 45, 30, 26, 17, 36, 44, 12, 40, 18, 14, 42, 24, 16, 28, 1, 11, 8, 25, 15, 29, 2, 5, 35, 23, 33, 3, 4, 32]; % XG

% rng(s);

[rows, cols] = size(data_L45);
training_set = 0.7;
checking_set = 0.2;
testing_set = 0.1;
training_num = ceil(rows*training_set);
testing_num = ceil(rows*testing_set);
validaion_num = rows-training_num-testing_num;

% 特徵index
feaures_index = [2:6,8];

training_data = num_data_L45(n(1:training_num), feaures_index);
checking_data = num_data_L45(n(training_num+1:training_num+validaion_num), feaures_index);
testing_data = num_data_L45(n(training_num+validaion_num+1:end), feaures_index);

[rows_f, cols_f] = size(feaures_index);

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
title("ANFIS weight prediction (training data)")
xlabel("No") 
ylabel("Weight (g)") 

% checking data
figure
[numRows_check, numCols_check] = size(checking_data);
x_check = (1:numRows_check);
plot(x_check,checking_data(:,cols_f),"or",x_check, predict_checking_data,"*b")
legend("Training Data","ANFIS Output")
title("ANFIS weight prediction (ckecking data)")
xlabel("No") 
ylabel("Weight (g)") 
% testing data
figure
[numRows_test, numCols_test] = size(testing_data);
x_test = (1:numRows_test);
plot(x_test,testing_data(:,cols_f),"or",x_test,predict_testing_data,"*b")
legend("Training Data","ANFIS Output")
title("ANFIS weight prediction (testing data)")
xlabel("No") 
ylabel("Weight (g)") 


% training error
figure
plot((1:epoch), trainError, 'or')
legend("trainError")
title("Training Error")
xlabel("Epoch") 
ylabel("Error (g)") 

figure
plot((1:epoch), chkError, 'b*')
legend("chkError")
title("Check Error")
xlabel("Epoch") 
ylabel("Error (g)") 


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

