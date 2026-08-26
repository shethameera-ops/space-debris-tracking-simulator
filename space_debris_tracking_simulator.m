%% ========================================================================
%  SPACE DEBRIS TRACKING SIMULATOR
%  Semester 6 Project - LEO Debris Simulation & Magnetic Capture Concept
%
%  Modules:
%   1. Orbital motion & state prediction        (two-body propagation)
%   2. Velocity and collision assessment         (relative velocity, TCA/miss distance)
%   3. Magnetic field-based debris capture sim   (dipole attraction model)
%   4. Material classification of debris         (RCS + magnetic susceptibility)
%   5. Deorbit and disposal modeling             (atmospheric drag decay)
% ========================================================================

clear; clc; close all;
rng(7);   % reproducible debris field

%% ------------------------- 1. CONSTANTS ---------------------------------
mu_E   = 398600.4418;      % Earth's gravitational parameter [km^3/s^2]
R_E    = 6378.137;         % Earth radius [km]
J2     = 1.08263e-3;       % Earth oblateness coefficient (unused, reserved)
w_E    = 7.2921159e-5;     % Earth rotation rate [rad/s]

N_debris   = 8;            % number of tracked debris objects
sim_time   = 6000;         % propagation duration [s]  (~1 orbit+)
dt         = 10;           % output time step [s]

%% ------------------------- 2. GENERATE DEBRIS FIELD ----------------------
% Random LEO orbital elements: altitude 300-1500 km, near-circular, varied inclination
debris = struct([]);
for k = 1:N_debris
    alt   = 300 + (1500-300)*rand();          % altitude [km]
    a     = R_E + alt;                        % semi-major axis [km]
    e     = 0.001 + 0.01*rand();              % near-circular eccentricity
    inc   = deg2rad(20 + 80*rand());          % inclination [rad]
    RAAN  = deg2rad(360*rand());
    argp  = deg2rad(360*rand());
    nu0   = deg2rad(360*rand());

    debris(k).a    = a;
    debris(k).e    = e;
    debris(k).inc  = inc;
    debris(k).RAAN = RAAN;
    debris(k).argp = argp;
    debris(k).nu0  = nu0;

    % --- synthetic sensor features for material classification ---
    % Radar cross-section [m^2] and magnetic susceptibility index [0-1]
    is_metal = rand() > 0.4;                  % ~60% metallic fragments
    if is_metal
        debris(k).RCS      = 0.01 + 0.5*rand();
        debris(k).mag_susc = 0.55 + 0.45*rand();
        debris(k).true_class = "Metallic";
    else
        debris(k).RCS      = 0.001 + 0.05*rand();
        debris(k).mag_susc = 0.05*rand();
        debris(k).true_class = "Non-metallic";
    end
end

%% ------------------------- 3. ORBITAL PROPAGATION (TWO-BODY) -------------
t_vec = 0:dt:sim_time;
Nt    = length(t_vec);

pos_hist = cell(N_debris,1);   % [Nt x 3] ECI position [km]
vel_hist = cell(N_debris,1);   % [Nt x 3] ECI velocity [km/s]
alt_hist = zeros(Nt, N_debris);
sma_hist = zeros(Nt, N_debris);

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);

for k = 1:N_debris
    [r0, v0] = oe2rv(debris(k).a, debris(k).e, debris(k).inc, ...
                      debris(k).RAAN, debris(k).argp, debris(k).nu0, mu_E);
    x0 = [r0; v0];

    [~, X] = ode45(@(t,x) twobody_eom(t,x,mu_E), t_vec, x0, opts);

    pos_hist{k} = X(:,1:3);
    vel_hist{k} = X(:,4:6);

    for i = 1:Nt
        r = norm(X(i,1:3));
        v = norm(X(i,4:6));
        alt_hist(i,k) = r - R_E;
        sma_hist(i,k) = 1/(2/r - v^2/mu_E);   % vis-viva -> instantaneous SMA
    end
end

fprintf('Propagated %d debris objects over %.0f s (dt = %.0f s)\n', ...
        N_debris, sim_time, dt);

%% ------------------------- 4. VELOCITY & COLLISION ASSESSMENT ------------
% Pairwise closest-approach screening across the propagated window
collision_thresh_km = 5;    % conjunction screening radius [km]
fprintf('\n--- Collision Screening (threshold = %.1f km) ---\n', collision_thresh_km);

conjunctions = [];
for i = 1:N_debris-1
    for j = i+1:N_debris
        rel_pos = pos_hist{i} - pos_hist{j};
        rel_vel = vel_hist{i} - vel_hist{j};
        dist    = vecnorm(rel_pos,2,2);
        relspd  = vecnorm(rel_vel,2,2);

        [min_dist, idx] = min(dist);
        if min_dist < collision_thresh_km
            conjunctions = [conjunctions; i, j, min_dist, relspd(idx), t_vec(idx)]; %#ok<AGROW>
            fprintf('  Debris %d <-> Debris %d : miss dist = %.2f km, rel. speed = %.2f km/s at t = %.0f s\n', ...
                     i, j, min_dist, relspd(idx), t_vec(idx));
        end
    end
end
if isempty(conjunctions)
    fprintf('  No conjunctions below threshold detected in this window.\n');
end

%% ------------------------- 5. MATERIAL CLASSIFICATION --------------------
% Simple threshold/nearest-centroid classifier on [RCS, magnetic susceptibility]
fprintf('\n--- Material Classification ---\n');
RCS_all  = [debris.RCS]';
susc_all = [debris.mag_susc]';
feat     = [RCS_all, susc_all];

% Centroids learned from the "metallic" vs "non-metallic" clusters generated above
metal_mask = susc_all > 0.4;
c_metal    = mean(feat(metal_mask,:),1);
c_nonmetal = mean(feat(~metal_mask,:),1);

pred_class = strings(N_debris,1);
for k = 1:N_debris
    d_metal    = norm(feat(k,:) - c_metal);
    d_nonmetal = norm(feat(k,:) - c_nonmetal);
    if d_metal < d_nonmetal
        pred_class(k) = "Metallic";
    else
        pred_class(k) = "Non-metallic";
    end
    debris(k).pred_class = pred_class(k);
    fprintf('  Debris %d: RCS=%.3f m^2, susc=%.2f -> classified as %s (true: %s)\n', ...
            k, debris(k).RCS, debris(k).mag_susc, pred_class(k), debris(k).true_class);
end

%% ------------------------- 6. MAGNETIC CAPTURE SIMULATION ----------------
% Conceptual magnetic dipole "tug" acting on metallic debris that enters
% a capture envelope around the chaser satellite (simplified point-dipole model).
fprintf('\n--- Magnetic Capture Simulation ---\n');

mu0        = 4*pi*1e-7;         % vacuum permeability
m_chaser   = 500;                % chaser magnetic dipole moment [A*m^2]
capture_R  = 50e-3;              % capture envelope radius [km] (50 m)
B_thresh   = 1e-6;               % capture-trigger field strength [T] (illustrative)

captured = [];
for k = 1:N_debris
    if debris(k).pred_class ~= "Metallic"
        continue
    end
    % use chaser = first debris object's trajectory as reference platform
    rel = pos_hist{k} - pos_hist{1};
    d   = vecnorm(rel,2,2);           % [km]
    d_m = d*1000;                     % [m]

    B = (mu0/(4*pi)) .* (2*m_chaser ./ d_m.^3);  % on-axis dipole field magnitude [T]

    within = d < capture_R & B > B_thresh;
    if any(within) && k ~= 1
        t_capture = t_vec(find(within,1));
        captured = [captured; k, t_capture]; %#ok<AGROW>
        fprintf('  Debris %d: METALLIC, entered capture envelope at t = %.0f s (B = %.2e T)\n', ...
                 k, t_capture, max(B(within)));
    else
        fprintf('  Debris %d: metallic but did not enter capture envelope this pass\n', k);
    end
end
if isempty(captured)
    fprintf('  No metallic debris captured in this simulation window.\n');
end

%% ------------------------- 7. DEORBIT / DISPOSAL MODELING ----------------
% Post-capture (or natural) orbital decay via atmospheric drag,
% using an exponential atmosphere model and averaged drag-decay ODE for 'a'.
fprintf('\n--- Deorbit Modeling (Atmospheric Drag Decay) ---\n');

Cd      = 2.2;                 % drag coefficient
AtoM    = 0.01;                % area-to-mass ratio [m^2/kg]
h0_ref  = 400;                 % reference altitude [km]
rho0    = 5.25e-13;            % density at reference altitude [kg/m^3] (approx, 400 km)
H_scale = 60;                  % scale height [km] (approx for 300-500 km band)

decay_days = 0:1:120;
a0_deorbit = R_E + 450;        % example: object starts at 450 km altitude
a_decay    = zeros(size(decay_days));
a_decay(1) = a0_deorbit;

for i = 2:length(decay_days)
    a_km   = a_decay(i-1);
    h      = a_km - R_E;
    rho    = rho0 * exp(-(h - h0_ref)/H_scale);     % [kg/m^3]
    n      = sqrt(mu_E/a_km^3);                      % mean motion [rad/s]
    v      = n*a_km;                                 % circular speed [km/s]

    % Averaged secular decay rate: da/dt = -Cd*A/m * rho * v * a  (King-Hele approx, simplified)
    da_dt_km_per_s = -Cd*AtoM*rho*(v*1000)*a_km / 1000;  % [km/s], rough scaling
    da_dt_km_per_day = da_dt_km_per_s * 86400;

    a_decay(i) = a_km + da_dt_km_per_day;
    if a_decay(i) <= R_E + 120   % re-entry interface ~120 km
        a_decay(i:end) = R_E + 120;
        fprintf('  Object reaches re-entry altitude (~120 km) after %.0f days\n', decay_days(i));
        break
    end
end
alt_decay = a_decay - R_E;

%% ============================ 8. PLOTS ====================================

% --- (a) 3D Earth + debris positions snapshot (matches review-video figure) ---
figure('Name','Orbit & Debris Field','Color','w');

subplot(1,2,1);
[xs,ys,zs] = sphere(40);
surf(xs*R_E, ys*R_E, zs*R_E, 'FaceColor',[0.2 0.5 0.9], 'EdgeColor','none'); hold on;
light('Position',[1 1 1]); lighting gouraud; material dull;
axis equal; grid on; view(35,20);
xlabel('X [km]'); ylabel('Y [km]'); zlabel('Z [km]');
title('LEO Debris Field (t = 0)');

colors = lines(N_debris);
for k = 1:N_debris
    plot3(pos_hist{k}(1,1), pos_hist{k}(1,2), pos_hist{k}(1,3), ...
          '.', 'Color', 'r', 'MarkerSize', 18);
    plot3(pos_hist{k}(:,1), pos_hist{k}(:,2), pos_hist{k}(:,3), ...
          '-', 'Color', [colors(k,:) 0.35], 'LineWidth', 0.75);
end
hold off;

% --- (b) Altitude vs time for all debris ---
subplot(1,2,2);
plot(t_vec/60, alt_hist, 'LineWidth', 1.1);
xlabel('Time [min]'); ylabel('Altitude [km]');
title('Altitude vs Time'); grid on;
legend(arrayfun(@(k) sprintf('Debris %d',k), 1:N_debris, 'UniformOutput', false), ...
       'Location','eastoutside');

% --- (c) Semi-major axis vs time ---
figure('Name','Semi-Major Axis History','Color','w');
plot(t_vec/60, sma_hist, 'LineWidth', 1.1);
xlabel('Time [min]'); ylabel('Semi-Major Axis [km]');
title('Semi-Major Axis vs Time'); grid on;
legend(arrayfun(@(k) sprintf('Debris %d',k), 1:N_debris, 'UniformOutput', false), ...
       'Location','eastoutside');

% --- (d) Deorbit decay curve ---
figure('Name','Deorbit Decay','Color','w');
plot(decay_days, alt_decay, 'LineWidth', 1.6, 'Color', [0.85 0.2 0.2]);
xlabel('Time [days]'); ylabel('Altitude [km]');
title('Atmospheric Drag Deorbit Decay (a_0 = 450 km alt.)');
grid on;

% --- (e) Material classification scatter ---
figure('Name','Material Classification','Color','w');
gscatter(RCS_all, susc_all, pred_class, [0.1 0.5 0.9; 0.9 0.3 0.1], 'o^', 8);
xlabel('Radar Cross-Section [m^2]'); ylabel('Magnetic Susceptibility Index');
title('Debris Material Classification'); grid on;

fprintf('\nSimulation complete. See generated figures for orbit field, altitude/SMA history,\ndeorbit decay curve, and material classification scatter.\n');

%% ========================== LOCAL FUNCTIONS ================================

function dxdt = twobody_eom(~, x, mu)
    % Two-body equations of motion: x = [rx ry rz vx vy vz]
    r = x(1:3);
    v = x(4:6);
    rnorm = norm(r);
    a = -mu * r / rnorm^3;
    dxdt = [v; a];
end

function [r_eci, v_eci] = oe2rv(a, e, inc, RAAN, argp, nu, mu)
    % Classical orbital elements -> ECI position/velocity vectors
    p = a*(1-e^2);
    r_pf = (p/(1+e*cos(nu))) * [cos(nu); sin(nu); 0];
    v_pf = sqrt(mu/p) * [-sin(nu); e+cos(nu); 0];

    R3_RAAN = [ cos(RAAN) -sin(RAAN) 0; sin(RAAN) cos(RAAN) 0; 0 0 1];
    R1_inc  = [1 0 0; 0 cos(inc) -sin(inc); 0 sin(inc) cos(inc)];
    R3_argp = [ cos(argp) -sin(argp) 0; sin(argp) cos(argp) 0; 0 0 1];

    Q = R3_RAAN * R1_inc * R3_argp;   % PQW -> ECI
    r_eci = Q * r_pf;
    v_eci = Q * v_pf;
end
