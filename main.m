% Load the data from the data set file
data = load("data.mat");

% Train and test using F1 (Steps 1, 2.1, 2.2)

f1_acc = univariateClassifier(data.F1);
f1_error = 1 - f1_acc;
% The accuracy was around 53% and the error rate was found to be 47%.

% Plot F2 vs F1
figure;
hold on;
title("Before normalization of F1");
xlabel("First Feature (F1)");
ylabel("Second Feature (F2)");
for i = 1 : 5
    plot(data.F1(:, i), data.F2(:, i), 'o');
end
legend('C1', 'C2', 'C3', 'C4', 'C5');

% Normalizing F1 rows (Step 3)

data.Z1 = (data.F1 - mean(data.F1, 2)) ./ std(data.F1, 0, 2);

% Plot F2 vs Z1
figure;
hold on;
title("After normalization of F1 to Z1");
xlabel("First Feature (Z1)");
ylabel("Second Feature (F2)");
for i = 1 : 5
    plot(data.Z1(:, i), data.F2(:, i), 'o');
end
legend('C1', 'C2', 'C3', 'C4', 'C5');

% Train and test using Z1 (Step 4.2)
z1_acc = univariateClassifier(data.Z1);
z1_error = 1 - z1_acc;
% The accuracy was around 88.31% and the error rate was found to be 11.69%.

% Train and test using F2 (Step 4.3)
f2_acc = univariateClassifier(data.F2);
f2_error = 1 - f2_acc;
% The accuracy was around 55.09% and the error rate was found to be 44.91%.

% Train and test using [Z1, F2](Step 4.4)
z1f2_acc = bivariateClassifier(data.Z1, data.F2); % 97.98% Accuracy
z1f2_error = 1 - z1f2_acc; % 2.02% Error Rate
% The accuracy was around 97.98% and the error rate was found to be 2.02%.