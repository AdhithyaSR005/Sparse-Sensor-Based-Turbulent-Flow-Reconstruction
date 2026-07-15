clc;
clear;
close all;

%% ============================================================
% LOAD ORIGINAL FULL DATA
%% ============================================================

data_resolved = load('u_final.mat');
u_final = data_resolved.empty;   % full time-resolved velocity field (88x88x12000)

%% ============================================================
% CREATE DATASETS
% Method: RANDOM CROP
%
% The sensor patch location is chosen randomly within the 88x88 domain.
% randi([1 68]) ensures the 21x21 window stays fully inside the domain
% (since 68 + 20 = 88, the last valid start index).
% This serves as a baseline: no domain knowledge is used to guide placement.
%% ============================================================

row_start = randi([1 68]);   % random row start (1 to 68, so window fits in domain)
col_start = randi([1 68]);   % random col start

% Extract the randomly placed 21x21 sensor patch across all time steps
time_resolved_crop = u_final( ...
    row_start:row_start+20,...
    col_start:col_start+20,...
    :);                               % size: 21x21x12000

% Sparse full field: full 88x88 domain sampled every 50th snapshot
sparse_fullfield = u_final(:,:,1:50:end); % size: 88x88x240

%% ============================================================
% PARAMETERS
%% ============================================================

trunk = 10;           % number of POD modes to retain after SVD (truncation level)
ps = 21*21;           % spatial points in sensor patch (441)
ps_full = 88*88;      % total spatial points in full domain (7744)

deltr = 49;           % time-delay window: each Ycol column stacks (deltr+1) = 50 snapshots
nt = 12000;           % total number of time snapshots in the time-resolved field
nn = 240;             % number of sparse snapshots (12000/50 = 240)

%% ============================================================
% SENSOR POD
% Processes the time-resolved sensor patch (21x21x12000).
% Goal: extract dominant spatial structures (POD modes) and
%       get a low-rank approximation of the sensor signal.
%% ============================================================

X_sensor = reshape(time_resolved_crop,[ps nt]);  % flatten spatial dims -> (441 x 12000)

mean_sensor = mean(X_sensor,2,'omitnan');         % spatial mean at each sensor point
X_sensor = X_sensor - mean_sensor;               % subtract mean to get fluctuating component

[U_sensor,S_sensor,V_sensor] = svds(X_sensor,trunk);  % SVD: U=spatial modes, S=energy, V=temporal

% Reconstruct a low-rank (truncated) version of the sensor data
X_sensor_truncated = U_sensor*S_sensor*V_sensor';

%% ============================================================
% SPARSE FULL FIELD POD
% Processes the non-time-resolved full field (88x88x240).
% Goal: extract POD modes of the full domain from sparse snapshots.
%       U_field and V_field are used in the EPOD reconstruction formula.
%% ============================================================

X_sparse = reshape(sparse_fullfield,[ps_full nn]);  % flatten -> (7744 x 240)

mean_sparse = mean(X_sparse,2,'omitnan');            % spatial mean of sparse field
X_sparse = X_sparse - mean_sparse;                  % subtract mean

[U_field,S_field,V_field] = svds(X_sparse,trunk);  % SVD of sparse field

% Low-rank reconstruction of sparse field
X_sparse_truncated = U_field*S_field*V_field';

%% ============================================================
% GROUND TRUTH (88x88x12000)
% Used as reference for computing reconstruction errors.
%% ============================================================

X_truth = reshape(u_final,[ps_full nt]);    % flatten -> (7744 x 12000)

mean_truth = mean(X_truth,2,'omitnan');      % spatial mean of the full time-resolved field
X_truth = X_truth - mean_truth;             % subtract mean

dod = max(abs(X_truth(:)));                 % max absolute value; used to normalize errors

%% ============================================================
% BUILD YCOL USING TIME MATCHING
%
% Ycol is the time-delay embedded sensor matrix.
% Each column corresponds to one sparse snapshot time and contains
% (deltr+1) = 50 consecutive sensor snapshots stacked together.
% Temporal embedding captures the dynamic history of the sensor signal,
% which improves correlation with the full field.
%% ============================================================

Ycol = zeros(ps*(deltr+1), nn, 'single'); % size: (441*50) x 240 = 22050 x 240

t1 = (0:nn-1)*50;   % timestamps of sparse full-field snapshots (0, 50, 100, ..., 11950)
t2 = (0:nt-1);      % timestamps of sensor data (0, 1, 2, ..., 11999)

for i = 1:nn

    % Find the sensor time index closest to the i-th sparse snapshot time
    i2tar = find(abs(t2 - t1(i)) == min(abs(t2 - t1(i))));
    i2tar = i2tar(1);

    st = i2tar;           % start index in sensor time series
    ed = i2tar + deltr;   % end index (50 consecutive snapshots)

    if ed > nt
        break;   % stop if window exceeds available sensor data
    end

    % Stack 50 consecutive sensor snapshots into one column of Ycol
    Ycol(:,i) = reshape( ...
        X_sensor_truncated(:,st:ed), ...
        [ps*(deltr+1),1]);

end

%% ============================================================
% SVD OF YCOL
% Decomposes the time-delay embedded sensor matrix.
% phipr: spatial basis of sensor embedding
% sumpr: singular values (energy scaling)
% psipr: temporal basis, used to build the correlation matrix with full field
%% ============================================================

[phipr,sumpr,psipr] = svd(Ycol,'econ');

%% ============================================================
% CORRELATION MATRIX (Xi / EEtr)
% EEtr = psipr' * V_field
% Captures the cross-correlation between the temporal modes of the
% sensor embedding (psipr) and the temporal modes of the full field (V_field).
% Entries with |EEtr| < 3/sqrt(nn) are zeroed out as they represent
% weak or noise-driven correlations that could hurt reconstruction.
%% ============================================================

EEtr = psipr' * V_field;
term = 3/sqrt(nn);  % noise threshold based on statistical significance

EEtr(abs(EEtr) < term) = 0;  % zero out weakly correlated entries

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
%   u_recon = U_field * EEtr' * pinv(sumpr) * phipr' * PP
%
% NOTE: S_field is ABSENT from this formula (unlike other scripts).
% This is an earlier version of the reconstruction; the correct
% formula should include S_field as: U_field * S_field * EEtr' * ...
%
% PP: time-delay embedded sensor vector for snapshot i (441*50 x 1)
%% ============================================================

nTest = 20;   % number of snapshots to reconstruct (set low for quick testing)
utrins = zeros(ps_full,nt-deltr,'single'); % pre-allocate: (7744 x 11951)

for i = 1:nTest%nt-deltr

    % Build the time-delay embedded sensor vector for snapshot i
    PP = reshape( ...
        X_sensor_truncated(:,i:i+deltr), ...
        [ps*(deltr+1),1]);  % size: (441*50) x 1 = 22050 x 1

    % Apply EPOD reconstruction formula (note: S_field omitted here)
   utrins(:,i) = U_field * S_field * EEtr' * pinv(sumpr) * phipr' * PP;

    fprintf('Completed %d of %d\n',i,nt-deltr);

end

%% ============================================================
% ERROR METRICS
%
%  error measures:

% - Error_imse: normalized RMSE across subsampled snapshots
% - ER        : RMS percentage error (main metric used in the paper)
%% ============================================================

truth_compare = X_truth(:,1:nt-deltr); % ground truth trimmed to match reconstruction length

% Subsample every 50th snapshot for aggregate error evaluation
A = utrins(:,1:50:end);
B = truth_compare(:,1:50:end);

mse_val = mean((A-B).^2,'all');     % mean squared error
Error_imse = sqrt(mse_val)/dod;     % normalized RMSE
err_ptrunc = rms(B-A,'all')/dod;    % alternative RMS error
ER = rms(((B-A)*100/dod),'all');    % main error metric: RMS percentage error

fprintf('Error_imse  = %.6f\n',Error_imse);
fprintf('ER          = %.4f %%\n',ER);

%% ============================================================
% VISUALIZATION
%% ============================================================

k = 1;

mean_field = reshape(mean_truth, 88, 88);

orig_snap  = reshape(truth_compare(:,k), 88, 88) + mean_field;
recon_snap = reshape(utrins(:,k), 88, 88) + mean_field;
error_snap = orig_snap - recon_snap;

figure

subplot(1,3,1)
imagesc(orig_snap)
axis equal tight
colorbar
title('Original Field')

subplot(1,3,2)
imagesc(recon_snap)
axis equal tight
colorbar
title('Reconstructed Field')

subplot(1,3,3)
imagesc(error_snap)
axis equal tight
colorbar
title('Error Field')

sgtitle(sprintf('EPOD Reconstruction (Snapshot %d)',k))

drawnow