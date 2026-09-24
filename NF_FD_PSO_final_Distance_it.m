% Example parameters
clc;clear;
MC_1x24=sparameters('data/Tx_1x24_RP_HFSSDesign1.s24p');
MC_1x24_3_5=MC_1x24.Parameters(:,:,2);
% Scattering Array/Monostatic Array SI channel
Tx_ind=[1:2:23];
Rx_ind=[2:2:24];
MC_SI_channel=MC_1x24_3_5(Rx_ind,Tx_ind);
MC_SI_channel_magdB_avg=mean(mean(20.*log10(abs(MC_SI_channel))));

if isempty(gcp('nocreate'))
    parpool;    % optional: start parallel pool
end

%%
fc     = 3.5e9;
c      = 3e8;
lambda = c / fc;
d      = 0.08;
num_elem=12;
Ni = num_elem; Nj = num_elem;

% Simple 1x12 ULAs on x-axis, separated by 1 m along z
x_i = ((0:Ni-1) - (Ni-1)/2).' * d;
x_j = ((0:Nj-1) - (Nj-1)/2).' * d;

% pos_i = [ zeros(Ni,1), x_i,zeros(Ni,1)];
pos_i = [ zeros(Ni,1), x_i-0.02,zeros(Ni,1)];
pos_j = [ zeros(Nj,1), x_j+0.02,zeros(Nj,1)];   % shifted by 1 m in z
%%
%% Desired DL/UL user locations (spherical: [r, theta_deg, phi_deg])
% DL_dis_set=[0.8:0.2:4, 5:3:20];
% UL_dis_set=[0.8:0.2:4, 5:3:20];
%% single example
DL_dis_set=[2.5];
UL_dis_set=[1];

for DL_count=1:length(DL_dis_set)
for UL_count=1:length(UL_dis_set)
desired_r_i   =  DL_dis_set(DL_count);
desired_r_j   = UL_dis_set(UL_count);
% desired_phi_i = 30;
% desired_phi_j =  30;
desired_phi_i = 0;
desired_phi_j =  -10;

user_i_desired = [desired_r_i, 0, desired_phi_i];   % [r, theta, phi]
user_j_desired = [desired_r_j, 0, desired_phi_j];

%% 1) Reference beams perfectly focused at desired user locations
[w_i_ref, w_j_ref] = FD_BF(pos_i, pos_j, user_i_desired, user_j_desired, fc);

% Reference beam gains at desired DL / UL user locations
Gi0    = beamGain(pos_i, w_i_ref, user_i_desired, fc);
Gj0    = beamGain(pos_j, w_j_ref, user_j_desired, fc);
Gi0_dB = 10*log10(Gi0 + eps);
Gj0_dB = 10*log10(Gj0 + eps);

%% 2) PSO variable definition: x = [r_i, phi_i, r_j, phi_j]
nVar = 4;

% % Search bounds (±distanceChange in r, ±angleChange in phi)
% angleChange    = 20;   % [deg]
% distanceChange = 0.5;   % [m]
% 
% r_i_min   = desired_r_i   - distanceChange;
% r_i_max   = desired_r_i   + distanceChange;
% phi_i_min = desired_phi_i - angleChange;
% phi_i_max = desired_phi_i + angleChange;
% 
% r_j_min   = desired_r_j   - distanceChange;
% r_j_max   = desired_r_j   + distanceChange;
% phi_j_min = desired_phi_j - angleChange;
% phi_j_max = desired_phi_j + angleChange;


% distanceChange = 5;   % [m]
angleChange    = 20;   % [deg]

r_i_min   = 0.5;
r_i_max   = 4* desired_r_i;
phi_i_min = desired_phi_i - angleChange;
phi_i_max = desired_phi_i + angleChange;

r_j_min   = 0.5;
r_j_max   = 4*desired_r_j;
phi_j_min = desired_phi_j - angleChange;
phi_j_max = desired_phi_j + angleChange;


lb = [r_i_min,   phi_i_min,   r_j_min,   phi_j_min];
ub = [r_i_max,   phi_i_max,   r_j_max,   phi_j_max];

% %% 3) PSO options (parallel enabled)
% options = optimoptions('particleswarm', ...
%     'Display','iter', ...
%     'SwarmSize',500, ...
%     'MaxIterations',400, ...
%     'UseParallel',true, ...
%     'UseVectorized',false);

%% 3) PSO options (parallel enabled)
options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'SwarmSize',800, ...
    'MaxIterations',400, ...
    'UseParallel',true, ...
    'UseVectorized',false);

%% 4) Objective handle
objHandle = @(x) psoObjectiveFD(x, pos_i, pos_j, fc, MC_SI_channel, ...
    desired_r_i, desired_phi_i, ...
    desired_r_j, desired_phi_j, ...
    Gi0_dB, Gj0_dB);

%% 5) Run PSO
% Sanity-check the Python RL-init PSO exported seed result directly.
% [x_opt, fval_opt] = particleswarm(objHandle, nVar, lb, ub, options);
% r_i_opt   = x_opt(1);
% phi_i_opt = x_opt(2);
% r_j_opt   = x_opt(3);
% phi_j_opt = x_opt(4);
r_i_opt   = 2.10973610731;
phi_i_opt = 0.43137961646;
r_j_opt   = 1.30008621432;
phi_j_opt = -11.5028210352;

user_i_sph_opt = [r_i_opt, 0, phi_i_opt];
user_j_sph_opt = [r_j_opt, 0, phi_j_opt];

[w_i_opt, w_j_opt] = FD_BF(pos_i, pos_j, user_i_sph_opt, user_j_sph_opt, fc);

[SI_lin_opt, SI_dB_opt] = SI_calculate(w_j_opt, w_i_opt, MC_SI_channel);
[SI_lin_ref, SI_dB_ref] = SI_calculate(w_j_ref, w_i_ref, MC_SI_channel);

fprintf('Optimal focus points:\n');
fprintf('Ref  DL: r_i = %.3f m, phi_i = %.2f deg\n', desired_r_i, desired_phi_i);
fprintf('Opt  DL: r_i = %.3f m, phi_i = %.2f deg\n', r_i_opt, phi_i_opt);

fprintf('Ref  UL: r_j = %.3f m, phi_j = %.2f deg\n', desired_r_j, desired_phi_j);
fprintf('Opt  UL: r_j = %.3f m, phi_j = %.2f deg\n', r_j_opt, phi_j_opt);

fprintf('Optimal SI (HFSS): %.2f dB\n', SI_dB_opt);
fprintf('Ref SI (HFSS): %.2f dB\n', SI_dB_ref);

%% Check DL beam gain at desired location (before/after optimization)
Gi_opt    = beamGain(pos_i, w_i_opt, user_i_desired, fc);
Gi_opt_dB = 10*log10(Gi_opt + eps);

fprintf('DL gain at desired user (ref) : %.2f dB\n', Gi0_dB);
fprintf('DL gain at desired user (opt) : %.2f dB\n', Gi_opt_dB);

Gj_opt    = beamGain(pos_j, w_j_opt, user_j_desired, fc);
Gj_opt_dB = 10*log10(Gj_opt + eps);

fprintf('UL gain at desired user (ref) : %.2f dB\n', Gj0_dB);
fprintf('UL gain at desired user (opt) : %.2f dB\n', Gj_opt_dB);


desired_r_i_set(UL_count,DL_count)=desired_r_i;
desired_r_j_set(UL_count,DL_count)=desired_r_j;
desired_phi_i_set(UL_count,DL_count)=desired_phi_i;
desired_phi_j_set(UL_count,DL_count)=desired_phi_j;

r_i_opt_set(UL_count,DL_count)=r_i_opt;
r_j_opt_set(UL_count,DL_count)=r_j_opt;
phi_i_opt_set(UL_count,DL_count)=phi_i_opt;
phi_j_opt_set(UL_count,DL_count)=phi_j_opt;

SI_dB_ref_set(UL_count,DL_count)=SI_dB_ref;
SI_dB_opt_set(UL_count,DL_count)=SI_dB_opt;
Gi0_dB_set(UL_count,DL_count)=Gi0_dB;
Gi_opt_dB_set(UL_count,DL_count)=Gi_opt_dB;

Gj0_dB_set(UL_count,DL_count)=Gj0_dB;
Gj_opt_dB_set(UL_count,DL_count)=Gj_opt_dB;

end
end
%%
%% Color maps of achievable SI (Ref vs Opt) over (DL angle, UL angle)

% Assuming:
%   DL_angle_set  : 1 x N_DL
%   UL_angle_set  : 1 x N_UL
%   SI_dB_ref_set : N_DL x N_UL  (rows = DL_count, cols = UL_count)
%   SI_dB_opt_set : N_DL x N_UL

DL_angles = DL_dis_set;
UL_angles = UL_dis_set;

% Use a common color scale for fair comparison
SI_min = min([SI_dB_ref_set(:); SI_dB_opt_set(:)]);
SI_max = max([SI_dB_ref_set(:); SI_dB_opt_set(:)]);

figure;

% --- Reference SI ---
subplot(1,2,1);
imagesc(UL_angles, DL_angles, SI_dB_ref_set);   % x = UL, y = DL
set(gca, 'YDir','normal');                     % so low DL angles at bottom
xlabel('UL angle \phi_j (deg)','FontSize',12);
ylabel('DL angle \phi_i (deg)','FontSize',12);
title('Reference SI (HFSS) [dB]','FontSize',14);
c = colorbar;
ylabel(c,'SI [dB]');
caxis([SI_min SI_max]);                        % same scale as OPT

% --- Optimized SI ---
subplot(1,2,2);
imagesc(UL_angles, DL_angles, SI_dB_opt_set);
set(gca, 'YDir','normal');
xlabel('UL angle \phi_j (deg)','FontSize',12);
ylabel('DL angle \phi_i (deg)','FontSize',12);
title('Optimized SI (HFSS) [dB]','FontSize',14);
c2 = colorbar;
ylabel(c2,'SI [dB]');
caxis([SI_min SI_max]);

set(gcf,'Color','w');

%%
SI_min = min([SI_dB_ref_set(:); SI_dB_opt_set(:)]);
SI_max = max([SI_dB_ref_set(:); SI_dB_opt_set(:)]);

figure;

% --- Reference SI ---
% subplot(1,2,1);
imagesc(UL_dis_set, DL_dis_set, SI_dB_ref_set);   % x = UL, y = DL
set(gca, 'YDir','normal');                     % so low DL angles at bottom
xlabel('UL angle \phi_j (deg)','FontSize',12);
ylabel('DL angle \phi_i (deg)','FontSize',12);
title('Reference SI (HFSS) [dB]','FontSize',14);
c = colorbar;
ylabel(c,'SI [dB]');
caxis([-50 -25]);
% caxis([SI_min SI_max]);                        % same scale as OPT

figure
% --- Optimized SI ---
% subplot(1,2,2);
imagesc(UL_dis_set, DL_dis_set, SI_dB_opt_set);
set(gca, 'YDir','normal');
xlabel('UL angle \phi_j (deg)','FontSize',12);
ylabel('DL angle \phi_i (deg)','FontSize',12);
title('Optimized SI (HFSS) [dB]','FontSize',14);
c2 = colorbar;
ylabel(c2,'SI [dB]');
caxis([-90 -40]);

set(gcf,'Color','w');

%%
figure;

% --- Reference SI ---
% subplot(1,2,1);
imagesc(UL_dis_set, DL_dis_set, SI_dB_ref_set);   % x = UL, y = DL
set(gca, 'YDir','normal','FontSize',20);                     % so low DL angles at bottom
xlabel('UL User Distance r_j (m)','FontSize',20);
ylabel('DL User Distance r_i (m)','FontSize',20);
% title('Reference SI (HFSS) [dB]','FontSize',14);
c = colorbar;
ylabel(c,'Self-Interference Level [dB]');
caxis([-100 -25]);
% caxis([SI_min SI_max]);                        % same scale as OPT
xticks([2:2:20])
yticks([2:2:20])

figure
% --- Optimized SI ---
% subplot(1,2,2);
imagesc(UL_dis_set, DL_dis_set, SI_dB_opt_set);
set(gca, 'YDir','normal','FontSize',20);
xlabel('UL User Distance r_j (m)','FontSize',20);
ylabel('DL User Distance r_i (m)','FontSize',20);
% title('Optimized SI (HFSS) [dB]','FontSize',14);
c2 = colorbar;
ylabel(c2,'Self-Interference Level [dB]');
% caxis([-90 -40]);
caxis([-100 -25]);

xticks([2:2:20])
yticks([2:2:20])


set(gcf,'Color','w');

a1=flip(SI_dB_ref_set,1);
a2=flip(SI_dB_opt_set,1);

%%
figure;
SI_improve = SI_dB_ref_set - SI_dB_opt_set;   % positive = improvement [dB]
imagesc(UL_angles, DL_angles, SI_improve);
set(gca,'YDir','normal');
xlabel('UL angle \phi_j (deg)','FontSize',12);
ylabel('DL angle \phi_i (deg)','FontSize',12);
title('SI improvement (Ref - Opt) [dB]','FontSize',14);
c = colorbar;
ylabel(c,'Improvement [dB]');
set(gcf,'Color','w');

%% φ-cut beam pattern for optimized DL beam
dis_ind = desired_r_i;
% Beamforming weights
w = conj(w_i_opt(:)); % N x 1 column vector
% Phi-scan definition
phi_scan = -90:1:90; % azimuth angle [deg]
theta_cut = 0; % elevation angle [deg]
% ANG format: [azimuth; elevation] = [phi; theta]
ang = [phi_scan; theta_cut * ones(size(phi_scan))];
% Array definition
N = length(w);
% sample array define
array = phased.ULA( ...
'NumElements', N, ...
'ElementSpacing', d);
% Create spherical-wavefront array response object
AR = phased.SphericalWavefrontArrayResponse( ...
'SensorArray', array, ...
'PropagationSpeed', 3e8, ...
'WeightsInputPort', true);
% Compute near-field spherical-wavefront response Output size is length(phi_scan) x 1
resp = AR(fc, ang, dis_ind, w);
% Convert to normalized power in dB
P = abs(resp).^2;
P_dB = pow2db(P ./ max(P));
% Plot normalized radiation pattern
figure;
plot(phi_scan, P_dB, 'LineWidth', 1.5);
xlabel('\phi (Azimuth angle, degrees)', 'FontSize', 16);
ylabel('Normalized Power Level (dB)', 'FontSize', 16);
title(sprintf('\\phi-cut at \\theta = %.1f^\\circ, r = %.2f m', theta_cut, dis_ind), ...
'FontSize', 16, ...
'Interpreter', 'tex');
grid on;
set(gca, 'FontSize', 16);
set(gcf, 'Color', 'white');
ylim([-40 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions
function cost = psoObjectiveFD(x, pos_i, pos_j, fc, MC_SI_channel, ...
    desired_r_i, desired_phi_i, ...
    desired_r_j, desired_phi_j, ...
    Gi0_dB, Gj0_dB)
% x = [r_i, phi_i, r_j, phi_j]

r_i   = x(1);
phi_i = x(2);
r_j   = x(3);
phi_j = x(4);

% Candidate focusing points (theta = 0)
user_i_sph = [r_i, 0, phi_i];
user_j_sph = [r_j, 0, phi_j];

% Beamforming weights for this candidate
[w_i, w_j] = FD_BF(pos_i, pos_j, user_i_sph, user_j_sph, fc);

% Self-interference from HFSS channel
[~, SI_dB] = SI_calculate(w_j, w_i, MC_SI_channel);  % dB

% Beam gains at the desired DL/UL users (fixed locations)
user_i_desired = [desired_r_i, 0, desired_phi_i];
user_j_desired = [desired_r_j, 0, desired_phi_j];

Gi = beamGain(pos_i, w_i, user_i_desired, fc);
Gj = beamGain(pos_j, w_j, user_j_desired, fc);

Gi_dB = 10*log10(Gi + eps);
Gj_dB = 10*log10(Gj + eps);

% Beam-gain variation relative to reference (focused at desired users)
var_i = Gi_dB - Gi0_dB;   % DL variation (should be >= -3 dB)
var_j = Gj_dB - Gj0_dB;   % UL variation (should be >= -3 dB)

% Penalty if beam gain drops more than 3 dB at desired users
drop_limit_dB = -3;   % allowed variation >= -3 dB

penalty = 0;
if var_i < drop_limit_dB
    penalty = penalty + (var_i - drop_limit_dB)^2;
end
if var_j < drop_limit_dB
    penalty = penalty + (var_j - drop_limit_dB)^2;
end

% Weight of the penalty relative to SI term
alpha = 50;   % tune if needed (bigger => stricter beam-gain constraint)

% Final cost (minimize). More negative SI_dB is better, but penalty pushes up.
cost = SI_dB + alpha * penalty;
end

function h = nearFieldChannelToUser(pos_elem, user_sph, k)
% nearFieldChannelToUser  Near-field channel normalized to center–user distance
%
%   pos_elem : N x 3 element positions [m]
%   user_sph : [r, theta_deg, phi_deg]  (θ = elevation, φ = azimuth)
%   k        : wavenumber = 2*pi/lambda
%
%   The channel is normalized w.r.t. the distance from the array
%   coordinate origin to the user, r0 = user_sph(1):
%       h_n = (r0 / dist_n) * exp(-j*k*(dist_n - r0))

% User spherical coordinates
r0    = user_sph(1);              % reference distance: center -> user
theta = deg2rad(user_sph(2));     % elevation
phi   = deg2rad(user_sph(3));     % azimuth

% User Cartesian position
[x_u, y_u, z_u] = sph2cart(phi, theta, r0);
user_cart = [x_u, y_u, z_u];

% Distances from each element to user
diff = pos_elem - user_cart;              % N x 3
dist = sqrt(sum(diff.^2, 2));             % N x 1

% Relative phase and amplitude normalized to r0
h = (r0 ./ dist) .* exp(-1j * k * (dist - r0));
end



function G = beamGain(pos_array, w, user_sph, fc)
% beamGain  Return |array response|^2 at a given user location (Tx viewpoint)
%
%   pos_array : N x 3 element positions
%   w         : N x 1 complex Tx weights
%   user_sph  : [r, theta_deg, phi_deg]
%   fc        : carrier frequency [Hz]

c      = 3e8;
lambda = c / fc;
k      = 2*pi / lambda;

h = nearFieldChannelToUser(pos_array, user_sph, k);  % N x 1
h = h(:);
w = w(:);

% Tx field at user: E = h^T w  (sum_n h_n * w_n)
E = h.' * w;

G = abs(E).^2;   % power gain
end


function [SI_dB,w_i,w_j] = beamLevelSI(pos_i, pos_j, user_i_sph, user_j_sph, fc)
% beamLevelSI  Estimate beam-level self-interference between sub-array i and j

c      = 3e8;
lambda = c / fc;
k      = 2*pi / lambda;

% Near-field channels from each sub-array to its own user
h_i_ui = nearFieldChannelToUser(pos_i, user_i_sph, k);  % Ni x 1
h_j_uj = nearFieldChannelToUser(pos_j, user_j_sph, k);  % Nj x 1

% Conjugate near-field beamforming (MDB-type)
w_i = conj(h_i_ui) / norm(h_i_ui);
w_j = conj(h_j_uj) / norm(h_j_uj);

% Inter-array near-field channel from sub-array i to j
H_ji = interArrayChannel(pos_i, pos_j, k);  % Nj x Ni

% Beam-level self-interference (scalar link gain)
g_SI   = w_j' * H_ji * w_i;
SI_lin = abs(g_SI).^2;
SI_dB  = 10*log10(SI_lin + eps);
end


function [w_i,w_j] = FD_BF(pos_i, pos_j, user_i_sph, user_j_sph, fc)
% FD_BF  Near-field MDB weights for DL (i) and UL (j) sub-arrays

c      = 3e8;
lambda = c / fc;
k      = 2*pi / lambda;

h_i_ui = nearFieldChannelToUser(pos_i, user_i_sph, k);  % Ni x 1
h_j_uj = nearFieldChannelToUser(pos_j, user_j_sph, k);  % Nj x 1

% Conjugate near-field beamforming (MDB)
w_i = conj(h_i_ui) / norm(h_i_ui);
w_j = conj(h_j_uj) / norm(h_j_uj);
end


function [SI_lin,SI_dB] = SI_calculate(w_j, w_i, H_ji)
% SI_calculate  Beam-level self-interference given weights and H_ji

g_SI   = w_j' * H_ji * w_i;
SI_lin = abs(g_SI).^2;
SI_dB  = 10*log10(SI_lin + eps);
end


function H_ji = interArrayChannel(pos_i, pos_j, k)
% interArrayChannel  Near-field channel matrix from sub-array i to j

Ni = size(pos_i, 1);
Nj = size(pos_j, 1);

H_ji = zeros(Nj, Ni);
for n = 1:Ni
    diff = pos_j - pos_i(n, :);      % Nj x 3
    dist = sqrt(sum(diff.^2, 2));    % Nj x 1
    H_ji(:, n) = exp(-1j * k * dist) ./ dist;
end
end

