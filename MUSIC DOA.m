% Direction of Arrival Estimation, i.e., Spatial Spectrum Estimation
%% MUSIC: Multiple Signal Classification
clear all; close all; %delete(findall(0, 'Type', 'figure'));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 100;    % Number of time snapshots
fs = 10^7;  % Sampling frequency
fc = 10^6;  % Center frequency of narrowband sources
M = 10;     % Number of array elements, i.e., sensors or antennas
N = 5;      % Number of sources
sVar = 1;   % Variance of the amplitude of the sources
% p snapshots of N narrowband sources with random amplitude of mean zero and covariance 1
s = sqrt(sVar)*randn(N, p).*exp(1i*(2*pi*fc*repmat([1:p]/fs, N, 1)));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
doa = [20; 50; 85; 110; 145];  %DOAs
cSpeed = 3*10^8 ;              % Speed of light
dist = 150;                    % Sensors (i.e., antennas) spacing in meters
% Constructing the Steering matrix
A = zeros(M, N);
for k = 1:N
    A(:, k) = exp(-1i*2*pi*fc*dist*cosd(doa(k))*(1/cSpeed)*[0:M-1]'); 
end
noiseCoeff = 1;                         % Variance of added noise
x = A*s + sqrt(noiseCoeff)*randn(M, p); % Sensor signals
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
R = (x*x')/p; % Empirical covariance of the antenna data
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
% STEP d: Finding the noise subspace and estimating the DOAs %%%%%%%%%%%%%
[V, D] = eig(R);
noiseSub = V(:, 1:M-N); % Noise subspace of R

theta = 0:1:180;        %Peak search
a = zeros(M, length(theta));
res = zeros(length(theta), 1);
for i = 1:length(theta)
    a(:, i) = exp(-1i*2*pi*fc*dist*cosd(i)*(1/cSpeed)*[0:M-1]');
    res(i, 1) = 1/(norm(a(:, i)'*noiseSub).^2);
end
plot(res);
title("MUSIC算法实现DOA估计");
xlabel("入射角");
ylabel("空间谱");
hold on
[resSorted, orgInd] = sort(res, 'descend');
DOAs = orgInd(1:N, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MUSIC: Multiple Signal Classification
figure(2)
clear all;  %delete(findall(0, 'Type', 'figure'));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 100;    % Number of time snapshots
fs = 10^7;  % Sampling frequency
fc = 10^6;  % Center frequency of narrowband sources
M = [10,20,30];     % Number of array elements, i.e., sensors or antennas
N = 5;      % Number of sources
sVar = 1;   % Variance of the amplitude of the sources
% p snapshots of N narrowband sources with random amplitude of mean zero and covariance 1
s = sqrt(sVar)*randn(N, p).*exp(1i*(2*pi*fc*repmat([1:p]/fs, N, 1)));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
doa = [20; 50; 85; 110; 145];  %DOAs
cSpeed = 3*10^8 ;              % Speed of light
dist = 150;                    % Sensors (i.e., antennas) spacing in meters
% Constructing the Steering matrix
for Mindex=1:length(M)
A = zeros(M(Mindex), N);
for k = 1:N
    A(:, k) = exp(-1i*2*pi*fc*dist*cosd(doa(k))*(1/cSpeed)*[0:M(Mindex)-1]'); 
end
noiseCoeff = 1;                         % Variance of added noise
x = A*s + sqrt(noiseCoeff)*randn(M(Mindex), p); % Sensor signals
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
R = (x*x')/p; % Empirical covariance of the antenna data
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
% STEP d: Finding the noise subspace and estimating the DOAs %%%%%%%%%%%%%
[V, D] = eig(R);
noiseSub = V(:, 1:M(Mindex)-N); % Noise subspace of R

theta = 0:1:180;        %Peak search
a = zeros(M(Mindex), length(theta));
res = zeros(length(theta), 1);
for i = 1:length(theta)
    a(:, i) = exp(-1i*2*pi*fc*dist*cosd(i)*(1/cSpeed)*[0:M(Mindex)-1]');
    res(i, 1) = 1/(norm(a(:, i)'*noiseSub).^2);
end
plot(res);
title("MUSIC算法实现DOA估计");
xlabel("入射角");
ylabel("空间谱");
hold on
[resSorted, orgInd] = sort(res, 'descend');
DOAs = orgInd(1:N, 1);
end
legend("阵元数为10","阵元数为20","阵元数为30");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3)
%% MUSIC: Multiple Signal Classification
clear all; %delete(findall(0, 'Type', 'figure'));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = [100,200,300];    % Number of time snapshots
fs = 10^7;  % Sampling frequency
fc = 10^6;  % Center frequency of narrowband sources
M = 10;     % Number of array elements, i.e., sensors or antennas
N = 5;      % Number of sources
sVar = 1;   % Variance of the amplitude of the sources
% p snapshots of N narrowband sources with random amplitude of mean zero and covariance 1
for pindex=1:length(p)
s = sqrt(sVar)*randn(N, p(pindex)).*exp(1i*(2*pi*fc*repmat([1:p(pindex)]/fs, N, 1)));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
doa = [20; 50; 85; 110; 145];  %DOAs
cSpeed = 3*10^8 ;              % Speed of light
dist = 150;                    % Sensors (i.e., antennas) spacing in meters
% Constructing the Steering matrix
A = zeros(M, N);
for k = 1:N
    A(:, k) = exp(-1i*2*pi*fc*dist*cosd(doa(k))*(1/cSpeed)*[0:M-1]'); 
end
noiseCoeff = 1;                         % Variance of added noise
x = A*s + sqrt(noiseCoeff)*randn(M, p(pindex)); % Sensor signals
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
R = (x*x')/p(pindex); % Empirical covariance of the antenna data
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
% STEP d: Finding the noise subspace and estimating the DOAs %%%%%%%%%%%%%
[V, D] = eig(R);
noiseSub = V(:, 1:M-N); % Noise subspace of R

theta = 0:1:180;        %Peak search
a = zeros(M, length(theta));
res = zeros(length(theta), 1);
for i = 1:length(theta)
    a(:, i) = exp(-1i*2*pi*fc*dist*cosd(i)*(1/cSpeed)*[0:M-1]');
    res(i, 1) = 1/(norm(a(:, i)'*noiseSub).^2);
end
plot(res);
title("MUSIC算法实现DOA估计");
xlabel("入射角");
ylabel("空间谱");
hold on
[resSorted, orgInd] = sort(res, 'descend');
DOAs = orgInd(1:N, 1);
end
legend("快拍数为100","快拍数为200","快拍数为300");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4)
%% MUSIC: Multiple Signal Classification
clear all; %close all; %delete(findall(0, 'Type', 'figure'));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 100;    % Number of time snapshots
fs = 10^7;  % Sampling frequency
fc = 10^6;  % Center frequency of narrowband sources
M = 10;     % Number of array elements, i.e., sensors or antennas
N = 5;      % Number of sources
sVar = 1;   % Variance of the amplitude of the sources
% p snapshots of N narrowband sources with random amplitude of mean zero and covariance 1
s = sqrt(sVar)*randn(N, p).*exp(1i*(2*pi*fc*repmat([1:p]/fs, N, 1)));
% STEP a: Simulating the Narrowband Sources %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
doa = [20; 50; 85; 110; 145];  %DOAs
cSpeed = 3*10^8 ;              % Speed of light
dist = [100,150,250];                    % Sensors (i.e., antennas) spacing in meters
% Constructing the Steering matrix
A = zeros(M, N);
for dindex=1:length(dist)
for k = 1:N
    A(:, k) = exp(-1i*2*pi*fc*dist(dindex)*cosd(doa(k))*(1/cSpeed)*[0:M-1]'); 
end
noiseCoeff = 1;                         % Variance of added noise
x = A*s + sqrt(noiseCoeff)*randn(M, p); % Sensor signals
% STEP b: Mixing the sources and getting the sensor signals %%%%%%%%%%%%%%
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
R = (x*x')/p; % Empirical covariance of the antenna data
% STEP c: Estimating the covariance matrix of the sensor array %%%%%%%%%%%
% STEP d: Finding the noise subspace and estimating the DOAs %%%%%%%%%%%%%
[V, D] = eig(R);
noiseSub = V(:, 1:M-N); % Noise subspace of R

theta = 0:1:180;        %Peak search
a = zeros(M, length(theta));
res = zeros(length(theta), 1);
for i = 1:length(theta)
    a(:, i) = exp(-1i*2*pi*fc*dist(dindex)*cosd(i)*(1/cSpeed)*[0:M-1]');
    res(i, 1) = 1/(norm(a(:, i)'*noiseSub).^2);
end
plot(res);
title("MUSIC算法实现DOA估计");
xlabel("入射角");
ylabel("空间谱");
hold on
[resSorted, orgInd] = sort(res, 'descend');
DOAs = orgInd(1:N, 1);
end
legend("阵元间距为100","阵元间距为150","阵元间距为250");

