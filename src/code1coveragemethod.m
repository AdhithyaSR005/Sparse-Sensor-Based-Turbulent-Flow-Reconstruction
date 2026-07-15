clc;
clear;
close all;

%% ============================================================
% LOAD ORIGINAL FULL DATA
%% ============================================================

data_resolved = load('u_final.mat');
u_final = data_resolved.empty;        % load the full time-resolved velocity field (88x88x12000)

%% ============================================================
% PARAMETERS
%% ============================================================

trunk = 30;           % number of POD modes to retain after SVD (truncation level)
window_size = 21;
ps = window_size*window_size;   % number of spatial points in one sensor patch (21x21 = 441)
ps_full = 88*88;                % total spatial points in the full domain (7744)

deltr = 49;           % time-delay window length; each Ycol column stacks (deltr+1) = 50 consecutive snapshots
nt = 12000;           % total number of time snapshots in the time-resolved field
nn = 240;             % total number of snapshots in the non-time-resolved (sparse) field (12000/50 = 240)

[Nx, Ny, ~] = size(u_final);   % spatial dimensions: Nx = Ny = 88

num_modes = trunk;              % number of POD modes used during the energy scan
num_snapshots = nt;             % used for POD energy evaluation in coverage scan

%% ============================================================
% AUTOMATIC DOMAIN OPTIMIZATION
% Method: Uniform Overlapping Coverage + POD Energy Scan (EtaMap)
%
% IDEA: Instead of picking the sensor window randomly or manually,
% we scan every possible 21x21 window across the 88x88 domain.
% For each window, we compute how much turbulent kinetic energy
% (captured by POD singular values) it contains.
% The window with the highest POD energy is selected as the sensor.
% To avoid missing any part of the domain, we also track a "coverage map"
% and keep placing windows until every point is covered at least 3 times.
%% ============================================================

disp('Starting automatic domain optimization...')

coverage       = zeros(Nx, Ny); % tracks how many times each spatial point has been covered by a window
targetCoverage = 3;             % we want every point covered at least 3 times before stopping

EtaMap = zeros(Nx - window_size + 1, Ny - window_size + 1); % stores the POD energy score for each candidate window position

max_possible_windows = 10000;           % upper bound on how many windows we might place
selected_windows      = zeros(max_possible_windows, 2); % records the (row, col) start of each placed window
window_counter        = 1;              % counts how many windows have been placed so far

% Keep placing windows until every spatial point is covered at least 3 times
while min(coverage(:)) < targetCoverage

    % Find the spatial point that has been covered the fewest times so far
    [~, idx_cov] = min(coverage(:));
    [xc, yc]     = ind2sub(size(coverage), idx_cov);

    % Center a 21x21 window on that least-covered point,
    % clamped so the window does not go outside the 88x88 domain
    xstart = max(1, min(xc, Nx - window_size + 1));
    ystart = max(1, min(yc, Ny - window_size + 1));

    xend   = min(xstart + window_size - 1, Nx);
    yend   = min(ystart + window_size - 1, Ny);

    % Recorrect start indices if the end was clipped by the domain boundary
    xstart = xend - window_size + 1;
    ystart = yend - window_size + 1;

    % Extract this 21x21xnt window from the full field
    window = u_final(xstart:xend, ystart:yend, 1:num_snapshots);

    % Reshape to 2D: (441 x 12000) for SVD
    X = reshape(window, [window_size * window_size, num_snapshots]);

    % Compute the top POD modes of this window via SVD
    [~, S, ~] = svds(X, num_modes);
    sigma     = diag(S);  % singular values = square roots of energy in each mode

    % POD energy score = sum of squared singular values (total turbulent kinetic energy captured)
    EtaMap(xstart, ystart) = sum(sigma .^ 2);

    % Mark all points inside this window as "covered once more"
    coverage(xstart : xstart + window_size - 1, ...
             ystart : ystart + window_size - 1) = ...
    coverage(xstart : xstart + window_size - 1, ...
             ystart : ystart + window_size - 1) + 1;

    % Store this window's starting position
    selected_windows(window_counter, :) = [xstart, ystart];
    window_counter = window_counter + 1;

    fprintf('Minimum Coverage = %d\n', min(coverage(:)));

end

% Trim the pre-allocated array to only the windows actually placed
selected_windows = selected_windows(1:window_counter - 1, :);

% FIND THE OPTIMAL WINDOW
% Pick the window position that had the highest POD energy score
[maxval, idx]  = max(EtaMap(:));
[xbest, ybest] = ind2sub(size(EtaMap), idx);

fprintf('\nOptimal region found:\n');
fprintf('x = %d\n', xbest);
fprintf('y = %d\n', ybest);
fprintf('Maximum POD Energy = %.6f\n', maxval);

% VISUALIZE COVERAGE MAP
% Shows how many times each spatial point was covered during the scan
figure
imagesc(coverage)
colorbar
axis equal tight
title('Spatial Coverage Map')
xlabel('Y Direction')
ylabel('X Direction')

%% ============================================================
% CREATE DATASETS (using the EtaMap-optimal window as sensor region)
%% ============================================================

% Sensor patch: 21x21 spatial window at the best POD-energy location, all 12000 time steps
time_resolved_crop = u_final( ...
    xbest:xbest+window_size-1, ...
    ybest:ybest+window_size-1, ...
    :);                             % size: 21x21x12000

% Sparse full field: full 88x88 domain but sampled every 50th snapshot
sparse_fullfield = u_final(:,:,1:50:end); % size: 88x88x240

%% ============================================================
% SENSOR POD
% Processes the time-resolved sensor patch (21x21x12000)
% Goal: extract dominant spatial structures (POD modes) and
%       get a low-rank approximation of the sensor signal
%% ============================================================

X_sensor = reshape(time_resolved_crop,[ps nt]); % flatten spatial dims -> (441 x 12000)

mean_sensor = mean(X_sensor,2,'omitnan');        % compute spatial mean at each sensor point
X_sensor = X_sensor - mean_sensor;              % subtract mean to get fluctuating component

[U_sensor,S_sensor,V_sensor] = svds(X_sensor,trunk); % SVD: U=spatial modes, S=energy, V=temporal coefficients

% Reconstruct a low-rank (truncated) version of the sensor data
X_sensor_truncated = U_sensor*S_sensor*V_sensor';
X_sensor_truncated = single(X_sensor_truncated); % convert to single precision to save memory

%% ============================================================
% SPARSE FULL FIELD POD
% Processes the non-time-resolved full field (88x88x240)
% Goal: extract POD modes of the full domain from the sparse snapshots
%       U_field and V_field are used in the EPOD reconstruction formula
%% ============================================================

X_sparse = reshape(sparse_fullfield,[ps_full nn]); % flatten spatial dims -> (7744 x 240)

mean_sparse = mean(X_sparse,2,'omitnan');           % spatial mean of sparse field
X_sparse = X_sparse - mean_sparse;                 % subtract mean

[U_field,S_field,V_field] = svds(X_sparse,trunk);  % SVD of sparse field

% Low-rank reconstruction of the sparse field
X_sparse_truncated = U_field*S_field*V_field';
X_sparse_truncated = single(X_sparse_truncated);

%% ============================================================
% GROUND TRUTH
% The full 88x88x12000 field used as reference for error calculation
%% ============================================================

X_truth = reshape(u_final,[ps_full nt]); % flatten -> (7744 x 12000)

mean_truth = mean(X_truth,2,'omitnan');  % spatial mean of full time-resolved field
X_truth = X_truth - mean_truth;          % subtract mean

dod = max(abs(X_truth(:)));              % maximum absolute value used to normalize errors

%% ============================================================
% BUILD YCOL USING TIME MATCHING
%
% Ycol is the time-delay embedded sensor matrix.
% Each column of Ycol corresponds to one sparse snapshot time,
% and contains (deltr+1) = 50 consecutive sensor snapshots stacked together.
% This embedding captures temporal dynamics in the sensor signal,
% which is essential for correlating it with the full field.
%% ============================================================

Ycol = zeros(ps*(deltr+1), nn, 'single'); % size: (441*50) x 240 = 22050 x 240

t1 = (0:nn-1)*50;   % timestamps of the sparse full-field snapshots (0, 50, 100, ..., 11950)
t2 = (0:nt-1);      % timestamps of the time-resolved sensor data   (0, 1, 2, ..., 11999)

for i = 1:nn

    % Find the sensor time index closest to the i-th sparse snapshot time
    i2tar = find(abs(t2 - t1(i)) == min(abs(t2 - t1(i))));
    i2tar = i2tar(1);

    st = i2tar;           % start index in the sensor time series
    ed = i2tar + deltr;   % end index (st to st+49, giving 50 snapshots)

    if ed > nt
        break;   % stop if we've run past the end of the time-resolved data
    end

    % Stack the 50 consecutive sensor snapshots into one column of Ycol
    Ycol(:,i) = reshape( ...
        X_sensor_truncated(:,st:ed), ...
        [ps*(deltr+1),1]);

end

%% ============================================================
% SVD OF YCOL
% Decomposes the time-delay embedded sensor matrix.
% phipr: spatial basis of sensor embedding
% sumpr: singular values (energy scaling)
% psipr: temporal basis, used to build correlation with full field
%% ============================================================

[phipr,sumpr,psipr] = svd(Ycol,'econ');

%% ============================================================
% CORRELATION MATRIX (Xi / EEtr)
% EEtr = psipr' * V_field
% This captures the correlation between the temporal modes of the
% sensor embedding (psipr) and the temporal modes of the full field (V_field).
% It is the key bridge between sensor and full-field dynamics in EPOD.
% NOTE: No thresholding applied here (for diagnostic inspection).
%% ============================================================

EEtr = psipr' * V_field;
disp(EEtr)
% Visualize absolute correlation values to assess sensor-field coupling
figure; imagesc(abs(EEtr)); colorbar; title('abs(EEtr)')

%% ============================================================
% DIMENSION CHECK
% Prints sizes of all key matrices to verify consistency
%% ============================================================

disp('----------------------------------')
disp(size(phipr))
disp(size(sumpr))
disp(size(psipr))
disp(size(EEtr))
disp(size(U_field))
disp(size(S_field))
disp('----------------------------------')

fprintf('Ycol = %d x %d\n',size(Ycol));
fprintf('phipr = %d x %d\n',size(phipr));
fprintf('EEtr = %d x %d\n',size(EEtr));

%% ============================================================
% EPOD RECONSTRUCTION
%
% Formula (from Chen, Raiola & Discetti 2022):
%   u_recon = U_field * S_field * EEtr' * pinv(sumpr) * phipr' * PP
%
% PP     : time-delay embedded sensor vector for snapshot i (441*50 x 1)
% phipr' : projects PP onto sensor embedding modes
% pinv(sumpr): scales by inverse singular values
% EEtr'  : maps from sensor temporal space to full-field temporal space
% U_field * S_field: maps back to physical full-field space
%% ============================================================

nTest = 20;  % number of snapshots to reconstruct (set low for quick testing)
utrins = zeros(ps_full,nt-deltr,'single'); % pre-allocate reconstruction output: (7744 x 11951)

for i = 1:nTest

    % Build the time-delay embedded sensor vector for snapshot i
    PP = reshape( ...
        X_sensor_truncated(:,i:i+deltr), ...
        [ps*(deltr+1),1]);  % size: (441*50) x 1 = 22050 x 1

    % Apply EPOD reconstruction formula
    utrins(:,i) = ...
          U_field ...
        * S_field ...
        * EEtr' ...
        * pinv(sumpr) ...
        * phipr' ...
        * PP;

    fprintf('Completed %d of %d\n',i,nt-deltr);

end

%% ============================================================
% ERROR METRICS
%
% Three error measures are computed:
% - Error_DG  : relative Frobenius norm error on a single snapshot
% - Error_imse: normalized root-mean-square error across all snapshots
% - ER        : RMS percentage error (main metric used in the paper)
%% ============================================================

truth_compare = X_truth(:,1:nt-deltr); % ground truth trimmed to match reconstruction length

% Error on one specific snapshot (snapshot 11951) using Frobenius norm
Error_DG = ...
    norm(utrins(:,11951)-truth_compare(:,11951),'fro') ...
    *100/ ...
    norm(truth_compare(:,11951),'fro');

% Subsample every 50th snapshot for aggregate error evaluation
A = utrins(:,:);
B = truth_compare(:,:);

mse_val = mean((A-B).^2,'all');     % mean squared error across all points and snapshots
Error_imse = sqrt(mse_val)/dod;     % normalized RMSE (relative to max absolute value)

err_ptrunc = rms(B-A,'all')/dod;    % alternative RMS error measure

ER = rms(((B-A)*100/dod),'all');    % main error metric: RMS percentage error

fprintf('Error_DG    = %.4f %%\n',Error_DG);
fprintf('Error_imse  = %.6f\n',Error_imse);
fprintf('ER          = %.4f %%\n',ER);

%% ============================================================
% VISUALIZATION
% ============================================================

k = 1; % snapshot index

% Convert mean vector back to 88x88
mean_field = reshape(mean_truth, 88, 88);

% Add mean back to visualize the actual velocity field
orig = reshape(truth_compare(:,1+50*(k-1)), 88, 88) + mean_field;

recon = reshape(utrins(:,1+50*(k-1)), 88, 88) + mean_field;

err = orig - recon;

figure

subplot(1,3,1)
imagesc(orig)
axis equal tight
colorbar
title('Original Field')

subplot(1,3,2)
imagesc(recon)
axis equal tight
colorbar
title('Reconstructed Field')

subplot(1,3,3)
imagesc(err)
axis equal tight
colorbar
title('Error Field')

sgtitle(sprintf('Coverage Method - EPOD Reconstruction (Snapshot %d)',1+50*(k-1)))