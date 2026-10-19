"""
Synthetic-but-physical RC helicopter flight log generator (ArduPilot TradHeli style).

Writes .xlsx files in the same 9-sheet format as the real flight controller export
(RATE, IMU_0, IMU_1, RCIN, RCOU, VIBE_0, VIBE_1, GPS, ATT) so they drop straight
into fn.folder_cleaner().

How the CG shows up (and why it is NOT trivially visible):
  * A forward CG puts a nose-down moment W*x_cg on the airframe. To hover, the rotor
    disk must tilt aft relative to the shaft until the hub moment cancels it:
        beta_trim = -x_cg * T / (h*T + K_hub)
    The rate controller's I-term ends up holding a nose-up POut trim and the pilot
    ends up holding a slightly nose-down attitude.
  * The same signals are also moved by wind, airspeed blowback, horizontal-stab
    moments, pilot maneuvers, gusts, battery sag, sensor bias and noise. Wind and
    airspeed effects rotate with heading; the CG effect is fixed in the body frame.
  * Each flight's real CG also differs from its nominal station by a small amount
    (ballast placement), so labels are not perfectly exact.

Run:  python generate_realistic_flights.py
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

# ----------------------------------------------------------------------------- config
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Raw_data_realistic")
SEED = 2026
FIRST_FLIGHT_NUM = 300
CG_STATIONS = np.round(np.arange(-0.75, 0.75 + 1e-9, 0.15), 2)     # inches, + = forward
N_PER_STATION = 9                                                  # 11 * 9 = 99
EXTRA_STATION = 0.00                                               # 100th flight
WIND_SPEED = {"L": (0.5, 2.0), "M": (2.5, 4.5), "H": (5.0, 7.5)}   # m/s mean wind
CG_PLACEMENT_SIGMA_IN = 0.02                                       # true CG scatter

FS = 400                     # control/sim loop (Hz)
DT = 1.0 / FS
LOG_DIV = 8                  # 400/8  = 50 Hz  (ATT, RATE, IMU)
SLOW_DIV = 5                 # 50/5   = 10 Hz  (RCIN, RCOU, VIBE)
GPS_DIV = 10                 # 50/10  = 5 Hz   (GPS)
G = 9.80665
RHO = 1.18
D2R, R2D = np.pi / 180.0, 180.0 / np.pi

# airframe (same helicopter every flight)
H_HUB = 0.17          # m, rotor hub above CG
K_HUB = 10.0          # N*m/rad, flybarless head stiffness
A_CYC = 0.52          # rad disk tilt per unit ROut/POut
TAU_FLAP = 0.04       # s
IXX, IYY, IZZ = 0.035, 0.085, 0.070
C_ROLL, C_PITCH, C_YAW = 0.90, 1.00, 0.30     # rotor / tail damping
K_BLOWBACK = 0.010    # rad per m/s of airspeed
K_HSTAB = 0.004       # N*m per (m/s)^2 nose-down with forward airspeed
L_TAIL, H_TAIL = 0.65, 0.06
K_TAIL = 5.0          # N tail thrust per unit (YOut + 0.25)
K_TORQUE = 0.035      # m, main rotor torque = K_TORQUE * T
C_LIN, CDA = 0.15, 0.04
K_WEATHERVANE = 0.05  # N*m per m/s side airspeed
GROUND_ROLL, GROUND_PITCH = 1.2, -1.6        # deg, attitude sitting on the skids
ANGLE_MAX = 45.0      # deg at full stick (clipped to 30)

# ArduPilot-like controller gains
ANG_P = 4.5
RAT = {  # FF, P, I, D, IMAX
    "roll": (0.13, 0.08, 0.25, 0.002, 0.4),
    "pitch": (0.15, 0.08, 0.25, 0.002, 0.4),
    "yaw": (0.09, 0.18, 0.12, 0.000, 0.4),
}


def ou_step(x, tau, sigma, rng, shape):
    """One Ornstein-Uhlenbeck step (stationary std = sigma)."""
    a = np.exp(-DT / tau)
    return a * x + sigma * np.sqrt(1 - a * a) * rng.standard_normal(shape)


def lp(x, y, tau):
    return x + (y - x) * (DT / (tau + DT))


# ----------------------------------------------------------------------------- flight plan
def build_flight_table(rng):
    rows = []
    for st in CG_STATIONS:
        n = N_PER_STATION + (1 if np.isclose(st, EXTRA_STATION) else 0)
        winds = (["L", "M", "H"] * 4)[:n]
        for w in winds:
            rows.append((st, w))
    rng.shuffle(rows)
    N = len(rows)
    t = pd.DataFrame(rows, columns=["cg_nominal_in", "wind_level"])
    t["flight_num"] = FIRST_FLIGHT_NUM + np.arange(N)
    t["cg_true_in"] = t.cg_nominal_in + rng.normal(0, CG_PLACEMENT_SIGMA_IN, N)
    t["wind_speed"] = [rng.uniform(*WIND_SPEED[w]) for w in t.wind_level]
    t["wind_dir_deg"] = rng.uniform(0, 360, N)
    t["mass_kg"] = 1.95 + rng.normal(0, 0.015, N)
    t["hover_coll"] = rng.uniform(0.50, 0.58, N)
    t["flight_time_s"] = rng.uniform(55, 110, N)        # takeoff -> start of landing
    t["t_interlock"] = rng.uniform(1.5, 3.0, N)
    t["t_spool"] = rng.uniform(4.0, 6.0, N)
    t["pilot_aggr"] = rng.uniform(0.8, 1.5, N)          # pilot style / maneuver size
    t["ground_heading"] = rng.uniform(0, 360, N)
    t["ground_slope_r"] = rng.normal(0, 0.4, N)         # landing spot not flat
    t["ground_slope_p"] = rng.normal(0, 0.4, N)

    def cg_code(x):
        if np.isclose(x, 0):
            return "0.00M"
        return f"{abs(x):.2f}{'F' if x > 0 else 'A'}"

    t["filename"] = [f"{n}B_{cg_code(c)}_{w}.xlsx" for n, c, w in zip(t.flight_num, t.cg_nominal_in, t.wind_level)]
    return t


# ----------------------------------------------------------------------------- simulation
def simulate(tab, rng):
    N = len(tab)
    m = tab.mass_kg.values
    x_cg = tab.cg_true_in.values * 0.0254
    t_inter = tab.t_interlock.values
    t_spool_end = t_inter + tab.t_spool.values
    t_takeoff_cmd = t_spool_end + 0.3
    t_land_cmd = t_takeoff_cmd + 2.5 + tab.flight_time_s.values
    aggr = tab.pilot_aggr.values
    hover_coll = tab.hover_coll.values
    w_spd = tab.wind_speed.values
    w_dir = tab.wind_dir_deg.values * D2R
    w_mean = np.stack([w_spd * np.cos(w_dir), w_spd * np.sin(w_dir)])   # wind TO (N,E)

    t_max = float(t_land_cmd.max() + 20.0)
    n_steps = int(t_max * FS)
    n_steps -= n_steps % (LOG_DIV * GPS_DIV)
    n_log = n_steps // LOG_DIV

    g_roll = GROUND_ROLL + tab.ground_slope_r.values
    g_pitch = GROUND_PITCH + tab.ground_slope_p.values

    # state
    phi, th, psi = g_roll * D2R, g_pitch * D2R, tab.ground_heading.values * D2R
    p = q = r = np.zeros(N)
    pos = np.zeros((2, N)); vel = np.zeros((2, N))
    z = np.zeros(N); vz = np.zeros(N)                    # altitude (up), climb rate
    b_lon = np.zeros(N); b_lat = np.zeros(N)
    omega = np.zeros(N)                                  # rotor speed fraction
    landed = np.ones(N, bool)
    airborne_once = np.zeros(N, bool)
    touchdown = np.full(N, np.inf)

    # controller
    I_r = I_p = I_y = np.zeros(N)
    gf = np.zeros((3, N)); gf2 = np.zeros((3, N))        # gyro filters
    yaw_target = psi.copy()

    # pilot
    tgt = np.zeros((2, N)); tgt_v = np.zeros((2, N)); tgt_v_cmd = np.zeros((2, N))
    pil_int = np.zeros((2, N)); pil_acc = np.zeros((2, N))
    tremor = np.zeros((3, N))
    seg_end = np.zeros(N); mode = np.zeros(N, int)
    circ_w = np.zeros(N); goal = np.zeros((2, N)); seg_spd = np.zeros(N)
    yaw_seg_end = np.zeros(N); yaw_rate_cmd = np.zeros(N); yaw_rate_s = np.zeros(N)
    alt_tgt = np.zeros(N); alt_seg_end = np.zeros(N); alt_goal = np.full(N, 3.0)
    coll = np.zeros(N)

    # disturbances
    turb = np.zeros((2, N)); turb_z = np.zeros(N); gust_m = np.zeros((3, N))
    gyro_n = np.zeros((3, N))

    keys = ["phi", "th", "psi", "phi_d", "th_d", "psi_d", "p", "q", "r", "pf", "qf", "rf",
            "RDes", "PDes", "YDes", "ROut", "POut", "YOut", "AOut", "fx", "fy", "fz", "az_up",
            "omega", "pn", "pe", "z", "vn", "ve", "vz", "s1", "s2", "s3", "s4", "landed"]
    L = {k: np.zeros((n_log, N), np.float32) for k in keys}

    for k in range(n_steps):
        t = k * DT

        # --- rotor speed (soft-start governor)
        spool = np.clip((t - t_inter) / (t_spool_end - t_inter), 0, 1)
        omega = np.where(t >= t_inter, 0.5 - 0.5 * np.cos(np.pi * spool), 0.0)

        # --- wind & disturbances (gustier in more wind)
        turb_sig = 0.15 + 0.22 * w_spd
        turb = ou_step(turb, 2.0, turb_sig, rng, (2, N))
        turb_z = ou_step(turb_z, 0.6, 0.05 + 0.06 * w_spd, rng, N)
        gust_m = ou_step(gust_m, 0.3, (0.02 + 0.012 * w_spd) * np.array([[1.0], [1.0], [0.6]]), rng, (3, N))
        wind = w_mean + turb

        # --- pilot: maneuver segments
        flying = ~landed
        free = flying & (t > t_takeoff_cmd + 8.0) & (t < t_land_cmd)
        new = free & (t >= seg_end)
        if new.any():
            idx = np.where(new)[0]
            u = rng.random(idx.size)
            mode[idx] = np.where(u < 0.35, 0, np.where(u < 0.8, 1, 2))
            seg_end[idx] = t + rng.uniform(3, 8, idx.size)
            # translate toward a random point inside a 15 m circle
            goal[:, idx] = rng.uniform(-1, 1, (2, idx.size)) * 15
            spd = rng.uniform(2.0, 9.0, idx.size) * aggr[idx]
            seg_spd[idx] = spd
            tgt_v_cmd[:, idx] = 0.0
            circ_w[idx] = np.where(mode[idx] == 2, rng.choice([-1, 1], idx.size) * spd / rng.uniform(5, 8, idx.size), 0)
            tgt_v_cmd[:, idx] = np.where(mode[idx] == 2, np.stack([np.zeros(idx.size), spd]), tgt_v_cmd[:, idx])
        hold = ~free
        tgt_v_cmd[:, hold] = 0.0
        mode[hold] = 0
        d_goal = goal - tgt
        dist = np.linalg.norm(d_goal, axis=0) + 1e-6
        v_go = d_goal / dist * np.minimum(seg_spd, 0.5 * dist)
        tgt_v_cmd = np.where(mode == 1, v_go, tgt_v_cmd)
        cw = np.where(mode == 2, circ_w, 0.0) * DT
        c, s = np.cos(cw), np.sin(cw)
        tgt_v_cmd = np.stack([c * tgt_v_cmd[0] - s * tgt_v_cmd[1], s * tgt_v_cmd[0] + c * tgt_v_cmd[1]])
        tgt_v_prev = tgt_v
        tgt_v = lp(tgt_v, tgt_v_cmd, 0.7)
        tgt_a = (tgt_v - tgt_v_prev) / DT
        tgt = np.where(flying, tgt + tgt_v * DT, pos)

        # yaw segments
        ny = free & (t >= yaw_seg_end)
        if ny.any():
            idx = np.where(ny)[0]
            u = rng.random(idx.size)
            sgn = rng.choice([-1, 1], idx.size)
            yaw_rate_cmd[idx] = np.where(u < 0.45, 0.0,
                                np.where(u < 0.85, sgn * rng.uniform(20, 90, idx.size),
                                         sgn * rng.uniform(90, 180, idx.size))) * aggr[idx]
            yaw_seg_end[idx] = t + rng.uniform(3, 8, idx.size)
        yaw_rate_cmd[~free] = 0.0
        yaw_rate_s = lp(yaw_rate_s, yaw_rate_cmd, 0.3)

        # altitude
        na = free & (t >= alt_seg_end)
        if na.any():
            idx = np.where(na)[0]
            alt_goal[idx] = rng.uniform(1.5, 7.0, idx.size)
            alt_seg_end[idx] = t + rng.uniform(5, 15, idx.size)
        goal_z = np.where(t < t_takeoff_cmd, -1.0,
                 np.where(t < t_takeoff_cmd + 8.0, 2.0,
                 np.where(t < t_land_cmd, alt_goal, -0.6)))
        alt_rate = np.where(t >= t_land_cmd, 0.7, 1.2)
        alt_tgt = np.where(landed & ~airborne_once, 0.0,
                           alt_tgt + np.clip(goal_z - alt_tgt, -alt_rate * DT, alt_rate * DT))

        # pilot position loop (with integral = learns trim / wind) -> desired tilt
        e = tgt - pos
        ev = tgt_v - vel
        pil_int = np.where(flying, np.clip(pil_int + e * DT, -15, 15), 0.0)
        acc_cmd = 1.3 * e + 2.0 * ev + 0.2 * pil_int + 1.5 * tgt_a
        pil_acc = lp(pil_acc, acc_cmd, 0.12)                     # human reaction lag
        cps, sps = np.cos(psi), np.sin(psi)
        a_fwd = cps * pil_acc[0] + sps * pil_acc[1]
        a_rgt = -sps * pil_acc[0] + cps * pil_acc[1]
        tremor = ou_step(tremor, 0.25, np.array([[0.035], [0.035], [0.03]]) * aggr, rng, (3, N))
        stick_r = np.clip(np.arctan2(a_rgt, G) * R2D / ANGLE_MAX + tremor[0], -1, 1)
        stick_p = np.clip(-np.arctan2(a_fwd, G) * R2D / ANGLE_MAX + tremor[1], -1, 1)
        stick_y = np.clip(yaw_rate_s / 200.0 + tremor[2] * flying, -1, 1)
        s1 = np.round(1500 + 500 * stick_r); s2 = np.round(1500 + 500 * stick_p)
        s4 = np.round(1500 + 500 * stick_y)

        # collective (pilot) from altitude error
        hov_now = hover_coll + 0.02 * np.clip((t - t_takeoff_cmd) / 60, 0, 2)      # battery sag
        az_cmd = np.clip(1.2 * (alt_tgt - z) - 1.6 * vz, -3, 3)
        cosb = np.cos(phi) * np.cos(th)
        T_air = m * (G + az_cmd) / np.maximum(cosb, 0.6)
        coll_air = hov_now + 0.35 * (T_air / (m * G) - 1)
        ramp = np.clip((t - t_spool_end) / 2.5, 0, 1)
        coll_ground_up = hov_now * 1.03 * ramp
        coll_after = np.clip(0.15 + (hov_now - 0.15) * (1 - (t - touchdown) / 1.5), 0.15, 1)
        coll_tgt = np.where(~airborne_once, coll_ground_up, np.where(np.isfinite(touchdown), coll_after, coll_air))
        coll = lp(coll, coll_tgt, 0.08)
        coll = np.where(omega <= 0, 0.0, coll)
        s3 = np.round(988 + 1024 * coll)
        aout = (s3 - 988) / 1024
        T = m * G * omega ** 2 * (1 + (aout - hov_now) / 0.35)
        T = np.maximum(T, 0)

        # takeoff / landing logic
        lift = landed & ~airborne_once & (T > m * G * 1.01)
        landed = landed & ~lift
        airborne_once |= lift
        td = (~landed) & (t > t_land_cmd) & (z <= 0) & (vz <= 0)
        touchdown = np.where(td & ~np.isfinite(touchdown), t, touchdown)
        landed |= td

        # --- attitude controller (stabilize)
        phi_d = np.clip((s1 - 1500) / 500 * ANGLE_MAX, -30, 30) * D2R
        th_d = np.clip((s2 - 1500) / 500 * ANGLE_MAX, -30, 30) * D2R
        yaw_target = np.where(landed, psi, yaw_target + (s4 - 1500) / 500 * 200 * D2R * DT)
        phi_d = np.where(landed, phi, phi_d); th_d = np.where(landed, th, th_d)
        yerr = (yaw_target - psi + np.pi) % (2 * np.pi) - np.pi
        dphi = np.clip(ANG_P * (phi_d - phi), -3.5, 3.5)
        dth = np.clip(ANG_P * (th_d - th), -3.5, 3.5)
        dpsi = np.clip(ANG_P * yerr, -3.5, 3.5) + (s4 - 1500) / 500 * 200 * D2R
        sphi, cphi, sth, cth = np.sin(phi), np.cos(phi), np.sin(th), np.cos(th)
        p_d = dphi - sth * dpsi
        q_d = cphi * dth + sphi * cth * dpsi
        r_d = -sphi * dth + cphi * cth * dpsi

        # measured rates (noise grows with rotor vibration) -> two filter stages
        gyro_n = ou_step(gyro_n, 0.004, 0.01 + 0.03 * omega ** 2, rng, (3, N))
        gm = np.stack([p, q, r]) + gyro_n
        gf = lp(gf, gm, 1 / (2 * np.pi * 20))
        gf2 = lp(gf2, gf, 1 / (2 * np.pi * 40))
        out = []
        for ax, (des, meas, Iax) in enumerate(((p_d, gf2[0], I_r), (q_d, gf2[1], I_p), (r_d, gf2[2], I_y))):
            FF, P, I, D, IMAX = RAT[("roll", "pitch", "yaw")[ax]]
            err = des - meas
            Iax = np.where(landed, Iax * 0.995, np.clip(Iax + I * err * DT, -IMAX, IMAX))
            o = np.clip(FF * des + P * err + Iax, -1, 1)
            out.append((o, Iax))
        (rout, I_r), (pout, I_p), (yout, I_y) = out
        rout = np.where(landed, rout * 0.2, rout)
        pout = np.where(landed, pout * 0.2, pout)

        # --- rotor & forces
        R11 = cth * cps; R12 = sphi * sth * cps - cphi * sps; R13 = cphi * sth * cps + sphi * sps
        R21 = cth * sps; R22 = sphi * sth * sps + cphi * cps; R23 = cphi * sth * sps - sphi * cps
        R31 = -sth;     R32 = sphi * cth;                    R33 = cphi * cth
        v_air_e = vel - wind
        u_air = R11 * v_air_e[0] + R21 * v_air_e[1]
        v_air = R12 * v_air_e[0] + R22 * v_air_e[1]
        b_lon = lp(b_lon, (-A_CYC * pout - K_BLOWBACK * u_air) * omega, TAU_FLAP)
        b_lat = lp(b_lat, (A_CYC * rout - K_BLOWBACK * v_air) * omega, TAU_FLAP)
        TT = K_TAIL * (yout + 0.25) * omega ** 2
        Fb = np.stack([T * np.sin(b_lon), T * np.sin(b_lat) - TT, -T * np.cos(b_lon) * np.cos(b_lat)])
        sp = np.hypot(v_air_e[0], v_air_e[1])
        Fd = -(C_LIN + 0.5 * RHO * CDA * sp) * v_air_e
        Fd_b = np.stack([R11 * Fd[0] + R21 * Fd[1], R12 * Fd[0] + R22 * Fd[1], R13 * Fd[0] + R23 * Fd[1]])
        F_body = Fb + Fd_b
        stiff = H_HUB * T + K_HUB * omega ** 2
        Mx = stiff * b_lat - H_TAIL * TT - C_ROLL * omega * p + gust_m[0]
        My = -stiff * b_lon - x_cg * T - C_PITCH * omega * q - K_HSTAB * u_air * np.abs(u_air) + gust_m[1]
        Mz = L_TAIL * TT - K_TORQUE * T - C_YAW * omega * r + K_WEATHERVANE * v_air + gust_m[2]

        # earth accel (NED), then up-positive vertical
        aN = (R11 * F_body[0] + R12 * F_body[1] + R13 * F_body[2]) / m
        aE = (R21 * F_body[0] + R22 * F_body[1] + R23 * F_body[2]) / m
        aD = (R31 * F_body[0] + R32 * F_body[1] + R33 * F_body[2]) / m + G + turb_z
        f_spec = F_body / m

        fly = ~landed
        p = np.where(fly, p + Mx / IXX * DT, 0.0)
        q = np.where(fly, q + My / IYY * DT, 0.0)
        r = np.where(fly, r + Mz / IZZ * DT, 0.0)
        phi_dot = p + (q * sphi + r * cphi) * np.tan(th)
        th_dot = q * cphi - r * sphi
        psi_dot = (q * sphi + r * cphi) / cth
        gr, gp = g_roll * D2R, g_pitch * D2R
        phi = np.where(fly, phi + phi_dot * DT, lp(phi, gr, 0.3))
        th = np.where(fly, th + th_dot * DT, lp(th, gp, 0.3))
        psi = np.where(fly, (psi + psi_dot * DT) % (2 * np.pi), psi)
        vel = np.where(fly, vel + np.stack([aN, aE]) * DT, 0.0)
        pos = pos + vel * DT
        vz = np.where(fly, vz - aD * DT, 0.0)
        z = np.where(fly, np.maximum(z + vz * DT, -0.01), 0.0)
        # on the ground the IMU sees only gravity
        g_spec = np.stack([-G * -np.sin(th), G * np.sin(phi) * np.cos(th) * -1, -G * np.cos(phi) * np.cos(th)])
        f_spec = np.where(fly, f_spec, g_spec)
        az_up = np.where(fly, -(aD - G) - G, 0.0)

        if k % LOG_DIV == 0:
            i = k // LOG_DIV
            for key, val in (("phi", phi), ("th", th), ("psi", psi), ("phi_d", phi_d), ("th_d", th_d),
                             ("psi_d", yaw_target % (2 * np.pi)), ("p", p), ("q", q), ("r", r),
                             ("pf", gf2[0]), ("qf", gf2[1]), ("rf", gf2[2]),
                             ("RDes", p_d), ("PDes", q_d), ("YDes", r_d),
                             ("ROut", rout), ("POut", pout), ("YOut", yout), ("AOut", aout),
                             ("fx", f_spec[0]), ("fy", f_spec[1]), ("fz", f_spec[2]), ("az_up", az_up),
                             ("omega", omega), ("pn", pos[0]), ("pe", pos[1]), ("z", z),
                             ("vn", vel[0]), ("ve", vel[1]), ("vz", vz),
                             ("s1", s1), ("s2", s2), ("s3", s3), ("s4", s4), ("landed", landed)):
                L[key][i] = val

    return L, touchdown


# ----------------------------------------------------------------------------- write one file
def write_flight(args):
    j, row, L_j, td = args
    rng = np.random.default_rng(SEED * 1000 + j)
    n_end = int(min((td + rng.uniform(1.5, 3.0)) * 50, len(L_j["phi"])))
    n_end -= n_end % SLOW_DIV
    d = {k: v[:n_end].astype(float) for k, v in L_j.items()}
    n = n_end
    om2 = d["omega"] ** 2

    t0 = int(rng.uniform(40e6, 120e6))
    epoch0 = row.epoch0

    def times(n_rows, step, offset, jitter):
        tu = t0 + offset + np.arange(n_rows) * step + rng.integers(-jitter, jitter + 1, n_rows)
        return tu.astype(np.int64), np.round(epoch0 + (tu - t0) / 1e6, 6)

    # ---- ATT (EKF attitude: small slowly varying estimation error)
    ekf_err = np.cumsum(rng.normal(0, 0.01, (2, n)), axis=1)
    ekf_err -= np.linspace(0, 1, n) * ekf_err[:, -1:]
    ekf_err = 0.12 * ekf_err / (np.abs(ekf_err).max() + 1e-9)
    tu, ep = times(n, 19998, 0, 4)
    att = pd.DataFrame({
        "TimeUS": tu,
        "DesRoll": np.round(d["phi_d"] * R2D, 2), "Roll": np.round(d["phi"] * R2D + ekf_err[0], 2),
        "DesPitch": np.round(d["th_d"] * R2D, 2), "Pitch": np.round(d["th"] * R2D + ekf_err[1], 2),
        "DesYaw": np.round(d["psi_d"] * R2D % 360, 2), "Yaw": np.round(d["psi"] * R2D % 360, 2),
        "ErrRP": np.round(np.clip(0.16 + 0.07 * np.sin(np.cumsum(rng.normal(0, 0.15, n))), 0, 0.4), 2),
        "ErrYaw": np.round(np.clip(0.5 + 0.27 * np.sin(np.cumsum(rng.normal(0, 0.1, n))), 0, 0.99), 2),
        "AEKF": 3, "Unix_Epoch_Time": ep})

    # ---- RATE
    tu, ep = times(n, 19998, 1904, 4)
    rate = pd.DataFrame({
        "TimeUS": tu,
        "RDes": d["RDes"] * R2D, "R": d["pf"] * R2D, "ROut": d["ROut"],
        "PDes": d["PDes"] * R2D, "P": d["qf"] * R2D, "POut": d["POut"],
        "YDes": d["YDes"] * R2D, "Y": d["rf"] * R2D, "YOut": d["YOut"],
        "ADes": 0, "A": d["az_up"] * 100 + rng.normal(0, 25 + 60 * om2, n),
        "AOut": np.round(d["AOut"], 3), "AOutSlew": 0, "Unix_Epoch_Time": ep})

    # ---- IMU_0 / IMU_1
    tvec = np.arange(n) / 50.0
    imus = []
    for inst, (ghz, ahz, vib_amp, scale) in enumerate(((2021, 2021, 0.30, 1.000), (2001, 2074, 0.42, 1.006))):
        gb = rng.normal(0, 0.002, 3); ab = rng.normal(0, 0.03, 3)
        f_alias = 6.7 + rng.normal(0, 0.3)                 # 1P rotor vibration aliased to 50 Hz log
        ph = rng.uniform(0, 2 * np.pi, 3)
        vib = vib_amp * om2 * np.sin(2 * np.pi * f_alias * tvec[None, :] + ph[:, None]) * np.array([[1], [1], [0.5]])
        gyr = np.stack([d["p"], d["q"], d["r"]]) + gb[:, None] + rng.normal(0, 1, (3, n)) * (0.012 + 0.035 * om2)
        acc = np.stack([d["fx"], d["fy"], d["fz"]]) * scale + ab[:, None] + vib \
            + rng.normal(0, 1, (3, n)) * (0.12 + np.array([[0.45], [0.45], [0.75]]) * om2)
        tu, ep = times(n - 1, 19998, 1947, 4)
        temp = np.round(44.0 + np.clip(np.cumsum(rng.normal(0, 0.002, n - 1)), 0, 0.6), 3)
        imus.append(pd.DataFrame({
            "TimeUS": tu, "I": inst,
            "GyrX": gyr[0, :-1], "GyrY": gyr[1, :-1], "GyrZ": gyr[2, :-1],
            "AccX": acc[0, :-1], "AccY": acc[1, :-1], "AccZ": acc[2, :-1],
            "EG": 0, "EA": 0, "T": temp, "GH": 1, "AH": 1, "GHz": ghz, "AHz": ahz, "Unix_Epoch_Time": ep}))

    # ---- 10 Hz sheets (sampled from the 50 Hz state)
    s = slice(0, n, SLOW_DIV)
    ns = n // SLOW_DIV
    tu_rc, ep_rc = times(ns, 99955, 80937, 30)
    interlock = d["omega"][s] > 0
    rcin = pd.DataFrame({"TimeUS": tu_rc,
                         "C1": d["s1"][s].astype(int), "C2": d["s2"][s].astype(int),
                         "C3": d["s3"][s].astype(int), "C4": d["s4"][s].astype(int),
                         "C5": 988, "C6": 1493 + (rng.random(ns) < 0.3).astype(int), "C7": 988,
                         "C8": np.where(interlock, 1500, 988)})
    for c in range(9, 15):
        rcin[f"C{c}"] = 1500
    rcin["Unix_Epoch_Time"] = ep_rc

    ro, po, yo, ao = d["ROut"][s], d["POut"][s], d["YOut"][s], d["AOut"][s]
    # H3-120 swashplate mix: servos at -60, +60, 180 deg
    base = (1400, 1587, 1573)
    roll_f = (0.866, -0.866, 0.0); pitch_f = (0.5, 0.5, -1.0)
    rcou = pd.DataFrame({"TimeUS": tu_rc + 12})
    for c in range(3):
        rcou[f"C{c+1}"] = np.round(base[c] + 170 * ao + 350 * (roll_f[c] * ro + pitch_f[c] * po)).astype(int)
    rcou["C4"] = np.round(1482 + 270 * yo).astype(int)
    for c in (5, 6, 7):
        rcou[f"C{c}"] = 0
    rcou["C8"] = np.where(interlock, 1500, 1100)
    for c in range(9, 15):
        rcou[f"C{c}"] = 0
    rcou["Unix_Epoch_Time"] = ep_rc + 12e-6

    vibes = []
    om2s = om2[s]
    spool_pass = (om2s > 0.15) & (om2s < 0.5)
    clip = np.cumsum(spool_pass & (rng.random(ns) < 0.15))
    for inst, basev in enumerate(((12.6, 22.6, 8.7), (15.5, 25.8, 7.3))):
        jit = 1 + 0.03 * np.sin(np.cumsum(rng.normal(0, 0.2, ns)))
        man = 1 + 0.05 * np.abs(d["q"][s] * 3)
        vals = [np.where(om2s > 0, b * om2s * jit * man + rng.normal(0, 0.15, ns) * om2s, rest)
                for b, rest in zip(basev, (0.03, 0.1, 0.03))]
        vibes.append(pd.DataFrame({"TimeUS": tu_rc + 22, "IMU": inst,
                                   "VibeX": np.maximum(vals[0], 0.03), "VibeY": np.maximum(vals[1], 0.1),
                                   "VibeZ": np.maximum(vals[2], 0.03), "Clip": clip.astype(int),
                                   "Unix_Epoch_Time": ep_rc + 22e-6}))

    # ---- GPS (5 Hz)
    g_ = slice(0, n, GPS_DIV)
    ng = len(range(0, n, GPS_DIV))
    tu_g, ep_g = times(ng, 199990, -2943, 2000)
    noise = np.cumsum(rng.normal(0, 0.03, (3, ng)), axis=1)
    pn, pe = d["pn"][g_] + noise[0], d["pe"][g_] + noise[1]
    pn -= pn[0]; pe -= pe[0]
    lat0, lng0 = row.home_lat, row.home_lng
    lat = lat0 + pn / 111_320.0
    lng = lng0 + pe / (111_320.0 * np.cos(lat0 * D2R))
    vn, ve = d["vn"][g_], d["ve"][g_]
    spd = np.hypot(vn, ve)
    crs = np.degrees(np.arctan2(ve, vn)) % 360
    crs = pd.Series(np.where(spd > 0.3, crs, np.nan)).ffill().fillna(rng.uniform(0, 360)).values
    gms0 = int(rng.uniform(1e8, 5e8)) // 200 * 200
    status = np.where(np.cumsum(rng.random(ng) < 0.01) % 2 == 0, 3, 4)
    nsats = np.clip(19 + np.round(np.cumsum(rng.normal(0, 0.08, ng))), 16, 22).astype(int)
    gps = pd.DataFrame({
        "TimeUS": tu_g, "I": 0, "Status": status, "GMS": gms0 + 200 * np.arange(ng), "GWk": 2440,
        "NSats": nsats, "HDop": rng.choice([0.5, 0.52, 0.52, 0.6], ng),
        "Lat": np.round(lat, 7), "Lng": np.round(lng, 7),
        "Alt": np.round(row.home_alt + d["z"][g_] + noise[2] * 0.5, 2),
        "Spd": np.round(np.where(spd < 0.05, 0, spd), 3), "GCrs": np.round(crs, 4),
        "VZ": np.round(-d["vz"][g_] + rng.normal(0, 0.05, ng), 3), "Yaw": 0, "U": 1,
        "Unix_Epoch_Time": ep_g, "lat_m": np.abs(pn), "lng_m": np.abs(pe)})

    sheets = {"RATE": rate, "IMU_0": imus[0], "IMU_1": imus[1], "RCIN": rcin, "RCOU": rcou,
              "VIBE_0": vibes[0], "VIBE_1": vibes[1], "GPS": gps, "ATT": att}
    path = os.path.join(OUT_DIR, row.filename)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    return row.filename, n / 50.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tab = build_flight_table(rng)
    # test sessions over a few days in Sept/Oct 2026, flights spaced by ~6-15 min
    day = np.repeat(np.arange(5), int(np.ceil(len(tab) / 5)))[:len(tab)]
    start = pd.Timestamp("2026-09-14 15:00", tz="America/Chicago").timestamp()
    tab["epoch0"] = start + day * 86400 * 3 + np.concatenate(
        [np.cumsum(rng.uniform(360, 900, (day == dd).sum())) for dd in range(5)])
    tab["home_lat"] = 32.46188 + rng.normal(0, 0.00008, len(tab))
    tab["home_lng"] = -94.72718 + rng.normal(0, 0.00008, len(tab))
    tab["home_alt"] = np.round(95.6 + rng.normal(0, 0.3, len(tab)), 2)

    print(f"Simulating {len(tab)} flights at {FS} Hz ...")
    L, touchdown = simulate(tab, rng)
    if not np.all(np.isfinite(touchdown)):
        raise RuntimeError(f"{(~np.isfinite(touchdown)).sum()} flights never landed")
    jobs = [(j, row, {k: v[:, j] for k, v in L.items()}, touchdown[j]) for j, row in tab.iterrows()]
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        for fname, dur in pool.imap_unordered(write_flight, jobs):
            print(f"  wrote {fname}  ({dur:.0f} s)")
    tab.drop(columns=["epoch0"]).to_csv(os.path.join(OUT_DIR, "_flight_manifest.csv"), index=False)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
