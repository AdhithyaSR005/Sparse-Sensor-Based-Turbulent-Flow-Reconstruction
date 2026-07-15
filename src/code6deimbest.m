clc; clear; close all; 

%% TUNING PARAMETERS 
debug_mode = true; % disable it during actual runs 
debug_nTest = 50; % only use this for debug mode 
trunk = 60;  % used for truncating the number of pods after performing svd              
deltr = 59;  % interval used for making non time resolved data                
use_psi_filter = false; 
spatial_smooth_sigma = 1.0;
window_size = 21; %required domain we need to crop  from 88x88
num_modes = 10; %same as trunk but used in DEIM 
%% ==============================================

fprintf('=== STEP 1: Loading Data ===\n');
data_resolved = load('u_final.mat'); % u_final.mat is the original full velocity field whose size is 88x88x12000
u_final = data_resolved.empty; % extracting the data
fprintf('STEP 1 DONE\n\n');

%% PARAMETERS
ps = window_size*window_size; % ps=21x21
ps_full = 7744; %ps_full is 88x88
nt = 12000; % timespans in time resolved field 
nn = 240; % timespans in non time resolved field since 12000/50 = 240 

[Nx, Ny, ~] = size(u_final);  % nx = ny = 88


%% DEIM / QR-PIVOTING SENSOR REGION SELECTION


fprintf('=== STEP 2: DEIM-based optimal sensor region ===\n');
disp('Computing global POD modes for DEIM scoring...')

% Need a reference POD basis over the FULL domain to rank points
X_full_sample = reshape(u_final(:,:,1:50:end), ps_full, nn); % it takes every 50th snapshot from 12000 in the 88x88 field , hence the size becomes 88x88x240 which is then reshaped to (88x88)x240
mean_full_sample = mean(X_full_sample, 2, 'omitnan'); %calculate the mean 
X_full_sample = X_full_sample - mean_full_sample; % subtract the mean to obtain the fluctuating part which contain the most dominant features 

[U_full, ~, ~] = svds(X_full_sample, num_modes);   % perform svd on the data and take the pods only 10 

disp('Running QR-pivoting (DEIM) on POD modes...')

% QR with column pivoting on U_full' selects the most independent ROWS
% (i.e., spatial points) of U_full
[~, ~, pvec] = qr(U_full', 'vector'); % perform qr decomposition 

% pvec is a ranking of all 7744 spatial points by DEIM importance,
% most important first
top_points = pvec(1:50);   % look at the top 50 most important points

% Convert linear indices back to (row,col) in the 88x88 domain
[top_rows, top_cols] = ind2sub([Nx Ny], top_points);

fprintf('Top 10 DEIM points (row,col):\n');
for q = 1:10
    fprintf('  (%d, %d)\n', top_rows(q), top_cols(q));
end

% Build a 2D map of DEIM importance (rank-based, for visualization)
deim_map = zeros(Nx, Ny);
for q = 1:length(pvec)
    [r,c] = ind2sub([Nx Ny], pvec(q));
    deim_map(r,c) = length(pvec) - q + 1;   % higher score = more important
end

figure
imagesc(deim_map)
axis equal tight
colorbar
hold on
plot(top_cols(1:10), top_rows(1:10), 'r*', 'MarkerSize', 10)
title('DEIM Importance Map (top 10 points marked)')


%% CONVERT TOP DEIM POINT TO A WINDOW (centered on most important point)


% Use the single most important DEIM point as the window center
center_row = top_rows(1);
center_col = top_cols(1);

half_w = floor(window_size/2);

row_start = max(1, min(center_row - half_w, Nx - window_size + 1)); % final 21 rows which has the best features  
col_start = max(1, min(center_col - half_w, Ny - window_size + 1)); % final 21 columns which has the best features 

fprintf('\nDEIM-centered region selected\n');
fprintf('Row start = %d\n', row_start);
fprintf('Col start = %d\n', col_start);
fprintf('Centered on DEIM point (%d, %d)\n', center_row, center_col);

fprintf('STEP 2 DONE\n\n');


%% STEP 3-4: PREPARING DATA & POD

fprintf('=== STEP 3-4: Preparing Data & POD ===\n');

time_resolved_crop = u_final(row_start:row_start+window_size-1, col_start:col_start+window_size-1, :); % creating the  21x21x12000 data -> time resolved data 
sparse_fullfield = u_final(:,:,1:50:end); % creating the 88x88x240 -> non time resolved 

% Sensor POD
X_sensor = reshape(time_resolved_crop, ps, nt); % 21x21x12000 reshape to (21x21)x12000
mean_sensor = mean(X_sensor,2,'omitnan'); % calculate mean  
X_sensor = X_sensor - mean_sensor; % subtract mean 
[U_sensor, S_sensor, V_sensor] = svds(X_sensor, trunk); %perform svd 
X_sensor_truncated = U_sensor*S_sensor*V_sensor';

% Field POD
X_sparse = reshape(sparse_fullfield, ps_full, nn); % 88x88x240 reshaped to (88x88)x240
mean_sparse = mean(X_sparse,2,'omitnan');  % calculate mean 
X_sparse = X_sparse - mean_sparse;   % subtract mean
[U_field, S_field, V_field] = svds(X_sparse, trunk); %perform svd
fprintf('PODs DONE\n\n');

%% BUILD YCOL
fprintf('=== STEP 5: Building Ycol ===\n');
Ycol = zeros(ps*(deltr+1), nn, 'single'); % creating a zero matrix of size ((441x50)x240) , this is the time delay data 
t1 = (0:nn-1)*50;
t2 = (0:nt-1);

for i = 1:nn
    [~, i2tar] = min(abs(t2 - t1(i))); % % Find closest sensor time to sparse field snapshot time
    st = i2tar;
    ed = min(st + deltr, nt);
    segment = X_sensor_truncated(:,st:ed);
    if size(segment,2) < deltr+1
        segment = [segment repmat(segment(:,end),1,deltr+1-size(segment,2))];
    end
    Ycol(:,i) = segment(:); %% Ycol: Time-delay embedded sensor matrix formed by stacking consecutive sparse measurements aligned with each full-field snapshot.
end
fprintf('STEP 5 DONE\n\n');

%% EPOD Reconstruction
fprintf('=== STEP 6: EPOD Reconstruction ===\n');
[phipr, sumpr, psipr] = svd(Ycol,'econ'); % perform svd on ycol  
EEtr = psipr' * V_field; % correlation matrix 
term = 3/sqrt(nn);
EEtr(abs(EEtr) < term) = 0; % data with low correlation is ignored 

if debug_mode
    nTest = min(debug_nTest, nt-deltr);
else
    nTest = nt - deltr;
end
utrins = zeros(ps_full, nTest, 'single'); % utrins is the reconstructed velocity field data whose size is 88x88x12000 
                                          %since it takes a lot of time so
                                          %for debugging instead of 12000,
                                          %im taking only less test cases
                                          % as given 

fprintf('Starting reconstruction...\n');
for i = 1:nTest
    st = i;
    ed = min(i + deltr, nt);
    PP = X_sensor_truncated(:,st:ed);
    if size(PP,2) < deltr+1
        PP = [PP repmat(PP(:,end),1,deltr+1-size(PP,2))];
    end
    PP = PP(:); %% PP: Embedded sensor history (current and previous sparse snapshots) for one test sample.
    
    utrins(:,i) = U_field * S_field * EEtr' * pinv(sumpr) * phipr' * PP; %same formula used in research paper 
    
    if mod(i,10)==0 || i==nTest
        fprintf('   Recon: %d / %d (%.1f%%)\n', i, nTest, (i/nTest)*100);
    end
end
fprintf('STEP 6 DONE\n\n');

%% ψ-Filtering + Spatial Smoothing
if use_psi_filter
    fprintf('=== STEP 7: Filtering ===\n');
    
    for p = 1:ps_full
        utrins(p,:) = movmean(double(utrins(p,:)), 11, 'endpoints','shrink');
    end
    
    [Xg, Yg] = meshgrid(-3:3, -3:3);
    gauss_kernel = exp(-(Xg.^2 + Yg.^2)/(2*spatial_smooth_sigma^2));
    gauss_kernel = gauss_kernel / sum(gauss_kernel(:));
    
    for i = 1:nTest
        snap = reshape(utrins(:,i), 88, 88);
        snap = conv2(snap, gauss_kernel, 'same');
        utrins(:,i) = snap(:);
    end
    fprintf('STEP 7 DONE\n\n');
end

%% ERROR CALCULATION
fprintf('=== STEP 9: Error ===\n');
X_truth = reshape(u_final, ps_full, nt);
mean_truth = mean(X_truth,2,'omitnan');
X_truth = X_truth - mean_truth;
dod = max(abs(X_truth(:)));

truth_compare = X_truth(:,1:nTest);
A_epod = utrins(:,1:50:end);
B = truth_compare(:,1:50:end);
ER_epod = rms(((B - A_epod)*100/dod),'all');

fprintf('\n=== FINAL RESULTS ===\n');
fprintf('EPOD ER = %.4f %%\n', ER_epod);

k = 1;

orig_snap  = reshape(truth_compare(:,k) + mean_truth, 88, 88);
recon_snap = reshape(utrins(:,k) + mean_truth, 88, 88);
error_snap = orig_snap - recon_snap;

figure('Position',[100 100 1200 400]);

subplot(1,3,1)
imagesc(orig_snap);
axis equal tight;
colorbar;
title('Original Field');
xlabel('x'); ylabel('y');

subplot(1,3,2)
imagesc(recon_snap);
axis equal tight;
colorbar;
title('EPOD Reconstructed Field');
xlabel('x'); ylabel('y');

subplot(1,3,3)
imagesc(error_snap);
axis equal tight;
colorbar;
title('Error Field');
xlabel('x'); ylabel('y');

sgtitle(sprintf('EPOD Reconstruction - Snapshot %d | ER = %.4f%%', k, ER_epod));

figure('Position',[100 100 900 600]);
for i = 1:4
    idx = 1 + (i-1)*10;
    subplot(2,2,i)
    imagesc(reshape(utrins(:,idx) + mean_truth, 88, 88));
    axis equal tight;
    colorbar;
    title(sprintf('Recon Snapshot %d', idx));
end
disp('Plots generated successfully!');