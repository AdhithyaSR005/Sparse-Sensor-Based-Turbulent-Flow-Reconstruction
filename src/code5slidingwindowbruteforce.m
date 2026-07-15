clc;
clear;
close all;

%% ============================================================
% PARAMETERS
%% ============================================================

trunk = 10;           % number of POD modes to retain after SVD (truncation level)
ps = 21*21;           % spatial points in sensor patch (441)
ps_full = 88*88;      % total spatial points in full domain (7744)

deltr = 49;           % time-delay window: each Ycol column stacks (deltr+1) = 50 snapshots
nt = 12000;           % total number of time snapshots in the time-resolved field
nn = 240;             % number of sparse snapshots (12000/50 = 240)

Nx = 88;
Ny = 88;

window_size = 21;
num_modes   = trunk;  % number of POD modes used in the energy scan

%% ============================================================
% LOAD ORIGINAL FULL DATA
%% ============================================================

data_resolved = load('u_final.mat');
u_final = data_resolved.empty;   % full time-resolved velocity field (88x88x12000)

%% ============================================================
% AUTOMATIC SENSOR DOMAIN OPTIMIZATION: Full Exhaustive POD Energy Scan
%
% Method: EXHAUSTIVE ETAMAP (all (68x68) = 4624 windows evaluated)
%
% IDEA: Scan EVERY possible 21x21 window across the 88x88 domain.
% For each window, compute how much turbulent kinetic energy its
% POD modes capture (sum of squared singular values = EtaMap score).
% The window with the highest score is selected as the sensor region.
%
% This is more thorough than Code 2 (which only covers the domain
% with a greedy coverage loop) -- here EVERY candidate window
% is evaluated, guaranteeing the globally optimal placement.
% The trade-off is that it is computationally expensive.
%% ============================================================

disp('Finding optimal sensor region...')

% Pre-allocate EtaMap: one entry per valid window starting position
% Valid range: rows 1 to (88-21+1)=68, cols 1 to 68
EtaMap = zeros(Nx-window_size+1, Ny-window_size+1);  % size: 68 x 68

for xstart = 1:Nx-window_size+1      % iterate over all valid row start positions
    for ystart = 1:Ny-window_size+1  % iterate over all valid col start positions

        xend = xstart + window_size - 1;
        yend = ystart + window_size - 1;

        % Extract the 21x21xnt data for this candidate window
        window = u_final(xstart:xend, ystart:yend, :);  % 21x21x12000

        % Flatten to 2D for SVD: (441 x 12000)
        X = reshape(window, [window_size*window_size, nt]);

        % Compute the top POD modes of this window via SVD
        [~,S,~] = svds(X, num_modes);

        sigma = diag(S);  % singular values = square roots of modal energy

        % POD energy score = sum of squared singular values (total captured TKE)
        EtaMap(xstart,ystart) = sum(sigma.^2);

    end
    fprintf('Row %d of %d complete\n', xstart, Nx-window_size+1);
end

% Find the window with the maximum POD energy score
[maxval,idx] = max(EtaMap(:));
[row_start,col_start] = ind2sub(size(EtaMap), idx);

fprintf('\nOptimal region found\n');
fprintf('Row start = %d\n', row_start);
fprintf('Col start = %d\n', col_start);
fprintf('Maximum POD Energy = %.6f\n', maxval);
fprintf('Number of nonzero entries in EtaMap = %d\n', nnz(EtaMap));

%% ============================================================
% VISUALIZATION OF ETAMAP
% Shows the POD energy score at every candidate window location.
% Brighter = higher energy = better sensor placement.
% The red star marks the selected optimal location.
%% ============================================================

figure
imagesc(EtaMap)
axis equal tight
colorbar
hold on
plot(col_start, row_start, 'r*', 'MarkerSize', 12)
title('POD Energy Map')

%% ============================================================
% CREATE DATASETS (using the exhaustive EtaMap-optimal window)
%% ============================================================

% Sensor patch: 21x21 at the globally optimal POD-energy location, all 12000 time steps
time_resolved_crop = u_final( ...
    row_start:row_start+20,...
    col_start:col_start+20,...
    :);                                    % size: 21x21x12000

% Sparse full field: full 88x88 domain, every 50th snapshot
sparse_fullfield = u_final(:,:,1:50:end);  % size: 88x88x240

%% ============================================================
% PARAMETERS (re-declared for clarity after dataset creation)
%% ============================================================

trunk = 10;
ps = 21*21;
ps_full = 88*88;
deltr = 49;
nt = 12000;
nn = 240;

%% ============================================================
% SENSOR POD
% Processes the time-resolved sensor patch (21x21x12000).
% Goal: extract dominant spatial structures (POD modes) and
%       get a low-rank approximation of the sensor signal.
%% ============================================================

X_sensor = reshape(time_resolved_crop,[ps nt]);  % flatten -> (441 x 12000)

mean_sensor = mean(X_sensor,2,'omitnan');         % spatial mean at each sensor point
X_sensor = X_sensor - mean_sensor;               % subtract mean

[U_sensor,S_sensor,V_sensor] = svds(X_sensor,trunk);  % SVD

% Low-rank reconstruction of sensor data
X_sensor_truncated = U_sensor*S_sensor*V_sensor';

X_sensor_truncated = single(X_sensor_truncated);  % convert to single precision to save memory

%% ============================================================
% SPARSE FULL FIELD POD
% Processes the non-time-resolved full field (88x88x240).
% U_field and V_field are used in the EPOD reconstruction formula.
%% ============================================================

X_sparse = reshape(sparse_fullfield,[ps_full nn]);  % flatten -> (7744 x 240)

mean_sparse = mean(X_sparse,2,'omitnan');            % spatial mean of sparse field
X_sparse = X_sparse - mean_sparse;                  % subtract mean

[U_field,S_field,V_field] = svds(X_sparse,trunk);  % SVD

% Low-rank reconstruction of sparse field
X_sparse_truncated = U_field*S_field*V_field';

X_sparse_truncated = single(X_sparse_truncated);  % convert to single precision

%% ============================================================
% GROUND TRUTH (88x88x12000)
%% ============================================================

X_truth = reshape(u_final,[ps_full nt]);    % flatten -> (7744 x 12000)

mean_truth = mean(X_truth,2,'omitnan');      % spatial mean of full time-resolved field
X_truth = X_truth - mean_truth;             % subtract mean

dod = max(abs(X_truth(:)));                 % max absolute value for error normalization

%% ============================================================
% BUILD YCOL USING TIME MATCHING
%
% Ycol is the time-delay embedded sensor matrix.
% Each column corresponds to one sparse snapshot time and contains
% (deltr+1) = 50 consecutive sensor snapshots stacked together.
%% ============================================================

Ycol = zeros(ps*(deltr+1), nn, 'single'); % size: (441*50) x 240 = 22050 x 240

t1 = (0:nn-1)*50;   % timestamps of sparse full-field snapshots (0, 50, ..., 11950)
t2 = (0:nt-1);      % timestamps of sensor data (0, 1, ..., 11999)

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
%
% NOTE: threshold term = 0 here, meaning NO entries are zeroed out.
% All correlation values are kept, including weak ones.
% This is a diagnostic setting to examine the full correlation structure.
% Diagnostic prints below compare Frobenius norms before/after thresholding.
%% ============================================================

EEtr = psipr' * V_field;
term = 0;   % threshold = 0: no filtering applied (keep all correlations)

EEtr(abs(EEtr) < term) = 0;  % (no effect when term = 0, kept for code consistency)

% Diagnostic: compare the full correlation matrix vs. the thresholded one
EEtr_before = psipr' * V_field;  % re-compute clean version for norm comparison

fprintf('EEtr Frobenius norm before threshold = %e\n', ...
    norm(EEtr_before,'fro'));

fprintf('EEtr Frobenius norm after threshold  = %e\n', ...
    norm(EEtr,'fro'));

fprintf('Nonzeros in EEtr = %d\n', nnz(EEtr));

% Print first 10 rows of EEtr and key singular values for inspection
disp(EEtr(1:10,:))
disp(diag(S_field))           % singular values of the full field (POD energy)
disp(diag(sumpr(1:10,1:10)))  % top singular values of Ycol embedding

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
%   u_recon = U_field * S_field * EEtr' * Sumpr_inv * phipr' * PP
%
% Sumpr_inv = diag(1./diag(sumpr)) replaces pinv(sumpr) for
% better numerical stability (avoids near-zero divisions).
%% ============================================================

nTest = 20;   % number of snapshots to reconstruct
utrins = zeros(ps_full,nt-deltr,'single'); % pre-allocate: (7744 x 11951)
Sumpr_inv = diag(1./diag(sumpr));          % stable inverse of diagonal singular value matrix

for i = 1:nTest

    % Build the time-delay embedded sensor vector for snapshot i
    PP = reshape( ...
        X_sensor_truncated(:,i:i+deltr), ...
        [ps*(deltr+1),1]);  % size: (441*50) x 1 = 22050 x 1

    % Apply EPOD reconstruction formula
    utrins(:,i) = ...
        U_field * S_field ...
        * EEtr' ...
        * Sumpr_inv ...
        * phipr' ...
        * PP;

    fprintf('Completed %d of %d\n',i,nt-deltr);

end

%% ============================================================
% ERROR METRICS
%
% Three error measures:
% - Error_DG  : relative Frobenius norm error on snapshot 11951
% - Error_imse: normalized RMSE across subsampled snapshots
% - ER        : RMS percentage error (main metric used in the paper)
%% ============================================================

truth_compare = X_truth(:,1:nt-deltr); % ground truth trimmed to match reconstruction length

% Error on a specific late snapshot using Frobenius norm
Error_DG = ...
    norm(utrins(:,11951)-truth_compare(:,11951),'fro') ...
    *100/ ...
    norm(truth_compare(:,11951),'fro');

% Subsample every 50th snapshot for aggregate error evaluation
A = utrins(:,1:50:end);
B = truth_compare(:,1:50:end);

mse_val = mean((A-B).^2,'all');     % mean squared error
Error_imse = sqrt(mse_val)/dod;     % normalized RMSE
err_ptrunc = rms(B-A,'all')/dod;    % alternative RMS error
ER = rms(((B-A)*100/dod),'all');    % main error metric: RMS percentage error

fprintf('Error_DG    = %.4f %%\n',Error_DG);
fprintf('Error_imse  = %.6f\n',Error_imse);
fprintf('ER          = %.4f %%\n',ER);

%% ============================================================
% VISUALIZATION
% Plots original, reconstructed, and error fields for snapshot k
%% ============================================================

k = 1;

% Convert the mean vector back to an 88x88 field
mean_field = reshape(mean_truth, 88, 88);

% Add the mean back to visualize the actual velocity field
orig = reshape(truth_compare(:,1+50*(k-1)), 88, 88) + mean_field;

recon = reshape(utrins(:,1+50*(k-1)), 88, 88) + mean_field;

% Error field
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

sgtitle(sprintf('EPOD Reconstruction (Snapshot %d)',1+50*(k-1)))

drawnow