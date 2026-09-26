"""Gymnasium environment for near-field full-duplex beam focusing."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .params import ENV_CONFIG
from .utils import make_ula_positions, near_field_channel_to_user, near_field_conjugate_weights


def extract_interleaved_tx_rx_hsi(Hsi_24):
    Hsi_24 = np.array(Hsi_24, dtype=np.complex128)
    if Hsi_24.shape != (24, 24):
        raise ValueError(f"Expected Hsi shape (24, 24), got {Hsi_24.shape}")
    tx_idx = np.arange(0, 24, 2)
    rx_idx = np.arange(1, 24, 2)
    return Hsi_24[np.ix_(rx_idx, tx_idx)], tx_idx, rx_idx


class NearFieldBeamFocusingEnv(gym.Env):
    """Near-field full-duplex Tx/Rx beamfocusing environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        Hsi,
        fc=ENV_CONFIG["fc"],
        element_spacing=ENV_CONFIG["element_spacing"],
        bwfn_deg=ENV_CONFIG["bwfn_deg"],
        max_steps=ENV_CONFIG["max_steps"],
        r_min=ENV_CONFIG["r_min"],
        r_max=ENV_CONFIG["r_max"],
        phi_min=ENV_CONFIG["phi_min"],
        phi_max=ENV_CONFIG["phi_max"],
        theta_min=ENV_CONFIG["theta_min"],
        theta_max=ENV_CONFIG["theta_max"],
        user_r_min=ENV_CONFIG["user_r_min"],
        user_r_max=ENV_CONFIG["user_r_max"],
        user_phi_min=ENV_CONFIG["user_phi_min"],
        user_phi_max=ENV_CONFIG["user_phi_max"],
        user_theta_min=ENV_CONFIG["user_theta_min"],
        user_theta_max=ENV_CONFIG["user_theta_max"],
        max_delta_r=ENV_CONFIG["max_delta_r"],
        max_delta_phi=ENV_CONFIG["max_delta_phi"],
        max_delta_theta=ENV_CONFIG["max_delta_theta"],
        init_focus_jitter_r=ENV_CONFIG["init_focus_jitter_r"],
        init_focus_jitter_phi=ENV_CONFIG["init_focus_jitter_phi"],
        init_focus_jitter_theta=ENV_CONFIG["init_focus_jitter_theta"],
        desired_r_dl=ENV_CONFIG["desired_r_dl"],
        desired_r_ul=ENV_CONFIG["desired_r_ul"],
        desired_theta_dl=ENV_CONFIG["desired_theta_dl"],
        desired_theta_ul=ENV_CONFIG["desired_theta_ul"],
        lambda_p=ENV_CONFIG["lambda_p"],
        lambda_boundary=ENV_CONFIG["lambda_boundary"],
        lambda_action=ENV_CONFIG["lambda_action"],
        lambda_degrade=ENV_CONFIG["lambda_degrade"],
        lambda_regret=ENV_CONFIG["lambda_regret"],
        lambda_si=ENV_CONFIG["lambda_si"],
        lambda_gain=ENV_CONFIG["lambda_gain"],
        lambda_dl_gain=ENV_CONFIG["lambda_dl_gain"],
        lambda_ul_gain=ENV_CONFIG["lambda_ul_gain"],
        min_gain_margin_db=ENV_CONFIG["min_gain_margin_db"],
        dl_gain_margin_db=ENV_CONFIG["dl_gain_margin_db"],
        ul_gain_margin_db=ENV_CONFIG["ul_gain_margin_db"],
        terminal_si_bonus=ENV_CONFIG["terminal_si_bonus"],
        direct_action_mode=ENV_CONFIG["direct_action_mode"],
        reward_mode=ENV_CONFIG["reward_mode"],
        reward_clip=ENV_CONFIG["reward_clip"],
        debug=False,
        seed=ENV_CONFIG["seed"],
    ):
        super().__init__()
        self.Hsi_full = np.asarray(Hsi, dtype=np.complex128)
        self.Hsi, self.tx_idx, self.rx_idx = extract_interleaved_tx_rx_hsi(self.Hsi_full)
        self.num_tx = self.Hsi.shape[1]
        self.num_rx = self.Hsi.shape[0]

        self.fc = float(fc)
        self.element_spacing = float(element_spacing)
        self.bwfn_deg = float(bwfn_deg)
        self.min_angular_sep = self.bwfn_deg / 2.0
        self.max_steps = int(max_steps)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.user_r_min = self.r_min if user_r_min is None else float(user_r_min)
        self.user_r_max = self.r_max if user_r_max is None else float(user_r_max)
        self.user_phi_min = self.phi_min if user_phi_min is None else float(user_phi_min)
        self.user_phi_max = self.phi_max if user_phi_max is None else float(user_phi_max)
        self.user_theta_min = self.theta_min if user_theta_min is None else float(user_theta_min)
        self.user_theta_max = self.theta_max if user_theta_max is None else float(user_theta_max)

        self.max_delta_r = float(max_delta_r)
        self.max_delta_phi = float(max_delta_phi)
        self.max_delta_theta = float(max_delta_theta)
        self.init_focus_jitter_r = float(init_focus_jitter_r)
        self.init_focus_jitter_phi = float(init_focus_jitter_phi)
        self.init_focus_jitter_theta = float(init_focus_jitter_theta)

        self.desired_r_dl = float(desired_r_dl)
        self.desired_r_ul = float(desired_r_ul)
        self.desired_theta_dl = float(desired_theta_dl)
        self.desired_theta_ul = float(desired_theta_ul)

        self.lambda_p = float(lambda_p)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_action = float(lambda_action)
        self.lambda_degrade = float(lambda_degrade)
        self.lambda_regret = float(lambda_regret)
        self.lambda_si = float(lambda_si)
        self.lambda_gain = float(lambda_gain)
        self.lambda_dl_gain = float(lambda_dl_gain)
        self.lambda_ul_gain = float(lambda_ul_gain)
        self.min_gain_margin_db = float(min_gain_margin_db)
        self.dl_gain_margin_db = self.min_gain_margin_db if dl_gain_margin_db is None else float(dl_gain_margin_db)
        self.ul_gain_margin_db = self.min_gain_margin_db if ul_gain_margin_db is None else float(ul_gain_margin_db)
        self.terminal_si_bonus = float(terminal_si_bonus)
        self.direct_action_mode = bool(direct_action_mode)
        self.reward_mode = str(reward_mode)
        self.reward_clip = reward_clip

        self.debug = bool(debug)
        self._rng = np.random.RandomState(seed)
        self.quadrant = 1

        self.r_dl_user = self.desired_r_dl
        self.r_ul_user = self.desired_r_ul
        self.phi_dl_user = 20.0
        self.phi_ul_user = 20.0
        self.theta_dl_user = self.desired_theta_dl
        self.theta_ul_user = self.desired_theta_ul

        self.r_tx_focus = None
        self.phi_tx_focus = None
        self.theta_tx_focus = None
        self.r_rx_focus = None
        self.phi_rx_focus = None
        self.theta_rx_focus = None
        self.current_SI_dB = 0.0
        self.prev_SI_dB = 0.0
        self.best_SI_dB = 0.0
        self.step_idx = 0

        full_pos = make_ula_positions(24, self.element_spacing / 2.0, axis="y")
        self.pos_tx = full_pos[self.tx_idx]
        self.pos_rx = full_pos[self.rx_idx]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(19, dtype=np.float32),
            high=np.ones(19, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_H(self):
        return self.Hsi.astype(np.complex128, copy=False)

    def _sample_users(self):
        while True:
            r_dl = self._rng.uniform(self.user_r_min, self.user_r_max)
            r_ul = self._rng.uniform(self.user_r_min, self.user_r_max)
            phi_dl = self._rng.uniform(self.user_phi_min, self.user_phi_max)
            phi_ul = self._rng.uniform(self.user_phi_min, self.user_phi_max)
            theta_dl = self._rng.uniform(self.user_theta_min, self.user_theta_max)
            theta_ul = self._rng.uniform(self.user_theta_min, self.user_theta_max)
            angular_sep = np.hypot(phi_dl - phi_ul, theta_dl - theta_ul)
            if angular_sep >= self.min_angular_sep:
                return (
                    float(r_dl),
                    float(r_ul),
                    float(phi_dl),
                    float(phi_ul),
                    float(theta_dl),
                    float(theta_ul),
                )

    def _sample_legal_focus_point(self, user_phi, user_theta):
        while True:
            r = self._rng.uniform(self.r_min, self.r_max)
            phi = self._rng.uniform(self.phi_min, self.phi_max)
            theta = self._rng.uniform(self.theta_min, self.theta_max)
            if abs(phi - user_phi) >= self.min_angular_sep:
                return float(r), float(phi), float(theta)

    def _sample_near_focus_point(self, base_r, base_phi, base_theta):
        r = base_r + self._rng.uniform(-self.init_focus_jitter_r, self.init_focus_jitter_r)
        phi = base_phi + self._rng.uniform(-self.init_focus_jitter_phi, self.init_focus_jitter_phi)
        theta = base_theta + self._rng.uniform(-self.init_focus_jitter_theta, self.init_focus_jitter_theta)
        return (
            float(np.clip(r, self.r_min, self.r_max)),
            float(np.clip(phi, self.phi_min, self.phi_max)),
            float(np.clip(theta, self.theta_min, self.theta_max)),
        )

    def _sample_focus_for_user(self, user_r, user_phi, user_theta, reset_mode):
        if reset_mode == "near_user":
            return self._sample_near_focus_point(user_r, user_phi, user_theta)
        raise ValueError("reset_mode must be 'near_user'")

    def compute_SI_dB(self, r_tx_focus, phi_tx_focus, theta_tx_focus, r_rx_focus, phi_rx_focus, theta_rx_focus):
        w_tx = near_field_conjugate_weights(self.pos_tx, r=r_tx_focus, phi_deg=phi_tx_focus, theta_deg=theta_tx_focus, fc=self.fc)
        w_rx = near_field_conjugate_weights(self.pos_rx, r=r_rx_focus, phi_deg=phi_rx_focus, theta_deg=theta_rx_focus, fc=self.fc)
        val = np.abs((w_rx.conj().T @ self._get_H()) @ w_tx)
        return 20.0 * np.log10(max(val, 1e-15))

    def compute_SI_reference_dB(self):
        return self.compute_SI_dB(
            self.r_dl_user,
            self.phi_dl_user,
            self.theta_dl_user,
            self.r_ul_user,
            self.phi_ul_user,
            self.theta_ul_user,
        )

    def compute_user_gain_dB(self, pos_elem, focus_sph, user_sph):
        w = near_field_conjugate_weights(
            pos_elem,
            r=focus_sph[0],
            phi_deg=focus_sph[1],
            theta_deg=focus_sph[2],
            fc=self.fc,
        )
        h = near_field_channel_to_user(
            pos_elem,
            np.array([user_sph[0], user_sph[2], user_sph[1]], dtype=np.float64),
            self.fc,
        )
        gain = np.abs(h @ w) ** 2
        return float(10.0 * np.log10(max(gain, 1e-15)))

    def compute_gain_metrics(self):
        tx_focus = (self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus)
        rx_focus = (self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus)
        dl_user = (self.r_dl_user, self.phi_dl_user, self.theta_dl_user)
        ul_user = (self.r_ul_user, self.phi_ul_user, self.theta_ul_user)
        tx_ref_focus = (self.r_dl_user, self.phi_dl_user, self.theta_dl_user)
        rx_ref_focus = (self.r_ul_user, self.phi_ul_user, self.theta_ul_user)

        dl_gain_ref = self.compute_user_gain_dB(self.pos_tx, tx_ref_focus, dl_user)
        dl_gain_now = self.compute_user_gain_dB(self.pos_tx, tx_focus, dl_user)
        ul_gain_ref = self.compute_user_gain_dB(self.pos_rx, rx_ref_focus, ul_user)
        ul_gain_now = self.compute_user_gain_dB(self.pos_rx, rx_focus, ul_user)
        return {
            "DL_gain_ref_dB": float(dl_gain_ref),
            "DL_gain_dB": float(dl_gain_now),
            "DL_gain_loss_dB": float(dl_gain_ref - dl_gain_now),
            "UL_gain_ref_dB": float(ul_gain_ref),
            "UL_gain_dB": float(ul_gain_now),
            "UL_gain_loss_dB": float(ul_gain_ref - ul_gain_now),
        }

    def compute_focus_penalty(self):
        tx_phi_error = abs(self.phi_tx_focus - self.phi_dl_user)
        rx_phi_error = abs(self.phi_rx_focus - self.phi_ul_user)
        tx_theta_error = abs(self.theta_tx_focus - self.theta_dl_user)
        rx_theta_error = abs(self.theta_rx_focus - self.theta_ul_user)
        tx_range_error = abs(self.r_tx_focus - self.r_dl_user)
        rx_range_error = abs(self.r_rx_focus - self.r_ul_user)
        return float(
            tx_phi_error / max(abs(self.phi_max - self.phi_min), 1e-12)
            + rx_phi_error / max(abs(self.phi_max - self.phi_min), 1e-12)
            + tx_theta_error / max(abs(self.theta_max - self.theta_min), 1e-12)
            + rx_theta_error / max(abs(self.theta_max - self.theta_min), 1e-12)
            + tx_range_error / max(abs(self.r_max - self.r_min), 1e-12)
            + rx_range_error / max(abs(self.r_max - self.r_min), 1e-12)
        )

    def _action_to_range(self, action_value):
        return self.r_min + 0.5 * (float(action_value) + 1.0) * (self.r_max - self.r_min)

    def _action_to_angle(self, action_value, min_value, max_value):
        return min_value + 0.5 * (float(action_value) + 1.0) * (max_value - min_value)

    def compute_boundary_penalty(self):
        r_margin = 0.10 * (self.r_max - self.r_min)
        theta_scale = max(abs(self.theta_min), abs(self.theta_max), 1e-12)

        def edge_penalty(value, lo, hi, margin):
            return max(0.0, (lo + margin - value) / margin) + max(0.0, (value - (hi - margin)) / margin)

        return float(
            edge_penalty(self.r_tx_focus, self.r_min, self.r_max, r_margin)
            + edge_penalty(self.r_rx_focus, self.r_min, self.r_max, r_margin)
            + abs(self.theta_tx_focus) / theta_scale
            + abs(self.theta_rx_focus) / theta_scale
        )

    def compute_reward(self, si_now, prev_si, best_si, si_ref, penalty, boundary_penalty, action_magnitude, gain_metrics, is_terminal):
        dl_gain_loss = max(0.0, gain_metrics["DL_gain_loss_dB"])
        ul_gain_loss = max(0.0, gain_metrics["UL_gain_loss_dB"])

        if self.reward_mode == "neg_si":
            reward = -si_now
        elif self.reward_mode == "neg_10_si":
            reward = -10.0 * si_now
        elif self.reward_mode == "neg_si_squared":
            reward = (-si_now) ** 2
        elif self.reward_mode == "neg_si_penalty":
            reward = -si_now - self.lambda_p * penalty
        elif self.reward_mode == "neg_si_gain_constraint":
            si_ref_advantage = np.clip(si_ref - si_now, -20.0, 20.0)
            si_step_advantage = np.clip(prev_si - si_now, -10.0, 10.0)
            si_best_advantage = np.clip(best_si - si_now, -10.0, 10.0)
            dl_excess_gain_loss = max(0.0, min(dl_gain_loss, 30.0) - self.dl_gain_margin_db)
            ul_excess_gain_loss = max(0.0, min(ul_gain_loss, 30.0) - self.ul_gain_margin_db)
            gain_constraint_penalty = (
                self.lambda_dl_gain * dl_excess_gain_loss / max(self.dl_gain_margin_db, 1e-12)
                + self.lambda_ul_gain * ul_excess_gain_loss / max(self.ul_gain_margin_db, 1e-12)
            )
            si_reward = 0.70 * si_ref_advantage + 0.20 * si_step_advantage + 0.10 * si_best_advantage
            if gain_constraint_penalty > 0.0:
                reward = (
                    -self.lambda_si * max(0.0, si_reward)
                    - self.lambda_gain * gain_constraint_penalty
                    - self.lambda_p * penalty
                )
            else:
                reward = 4.0 + self.lambda_si * si_reward - self.lambda_p * penalty
        elif self.reward_mode == "neg_10_si_penalty":
            reward = -10.0 * si_now - self.lambda_p * penalty
        elif self.reward_mode == "si_improvement":
            reward = prev_si - si_now
        elif self.reward_mode == "si_improvement_hybrid":
            reward = (prev_si - si_now) + 0.05 * (-si_now)
        elif self.reward_mode == "relative_si_hybrid":
            reward = (prev_si - si_now) + 0.20 * (si_ref - si_now)
        elif self.reward_mode == "sequential_hold":
            reward = 0.50 * (prev_si - si_now) + 0.20 * (si_ref - si_now) - 0.30 * max(0.0, si_now - best_si)
        elif self.reward_mode == "direct_si":
            reward = (si_ref - si_now) + 0.05 * (-si_now)
        elif self.reward_mode == "positive_si_gain":
            si_ref_improvement = np.clip(si_ref - si_now, -20.0, 20.0)
            si_step_improvement = np.clip(prev_si - si_now, -10.0, 10.0)
            dl_excess_gain_loss = max(0.0, min(dl_gain_loss, 30.0) - self.dl_gain_margin_db)
            ul_excess_gain_loss = max(0.0, min(ul_gain_loss, 30.0) - self.ul_gain_margin_db)
            gain_constraint_penalty = 0.50 * dl_excess_gain_loss + 2.00 * ul_excess_gain_loss
            reward = (
                4.0
                + 0.30 * si_ref_improvement
                + 0.20 * si_step_improvement
                - self.lambda_gain * gain_constraint_penalty / 3.0
                - 0.20 * penalty
            )
        else:
            raise ValueError(
                "reward_mode must be one of: neg_si, neg_10_si, neg_si_squared, "
                "neg_si_penalty, neg_si_gain_constraint, neg_10_si_penalty, "
                "si_improvement, si_improvement_hybrid, relative_si_hybrid, "
                "sequential_hold, direct_si, positive_si_gain"
            )

        if self.reward_mode == "neg_si_gain_constraint":
            reward -= 0.25 * self.lambda_boundary * boundary_penalty
            reward -= 0.25 * self.lambda_action * action_magnitude
        else:
            reward -= self.lambda_boundary * boundary_penalty
            reward -= self.lambda_action * action_magnitude
            reward -= self.lambda_degrade * max(0.0, si_now - prev_si)
            reward -= self.lambda_regret * max(0.0, si_now - best_si)
            if is_terminal:
                reward += self.terminal_si_bonus * (si_ref - min(best_si, si_now))

        if self.reward_clip is not None:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        return float(reward)

    def _norm_range(self, value):
        return 2.0 * (value - self.r_min) / max(self.r_max - self.r_min, 1e-12) - 1.0

    def _norm_angle(self, value, min_value, max_value):
        scale = max(abs(min_value), abs(max_value), 1e-12)
        return np.clip(value / scale, -1.0, 1.0)

    def _norm_si(self, value):
        return np.clip(value / 100.0, -1.0, 1.0)

    def _norm_gain_loss(self, value):
        return np.clip(value / 30.0, -1.0, 1.0)

    def _get_obs(self):
        gain_metrics = self.compute_gain_metrics()
        return np.array(
            [
                self._norm_range(self.r_tx_focus),
                self._norm_angle(self.phi_tx_focus, self.phi_min, self.phi_max),
                self._norm_angle(self.theta_tx_focus, self.theta_min, self.theta_max),
                self._norm_range(self.r_rx_focus),
                self._norm_angle(self.phi_rx_focus, self.phi_min, self.phi_max),
                self._norm_angle(self.theta_rx_focus, self.theta_min, self.theta_max),
                self._norm_range(self.r_dl_user),
                self._norm_angle(self.phi_dl_user, self.phi_min, self.phi_max),
                self._norm_angle(self.theta_dl_user, self.theta_min, self.theta_max),
                self._norm_range(self.r_ul_user),
                self._norm_angle(self.phi_ul_user, self.phi_min, self.phi_max),
                self._norm_angle(self.theta_ul_user, self.theta_min, self.theta_max),
                self._norm_si(self.current_SI_dB),
                self._norm_si(self.prev_SI_dB),
                self._norm_si(self.current_SI_dB - self.prev_SI_dB),
                self._norm_si(self.best_SI_dB),
                self._norm_si(self.current_SI_dB - self.best_SI_dB),
                self._norm_gain_loss(gain_metrics["DL_gain_loss_dB"]),
                self._norm_gain_loss(gain_metrics["UL_gain_loss_dB"]),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        options = {} if options is None else options
        reset_mode = options.get("reset_mode", "near_user")
        self.step_idx = 0
        (
            self.r_dl_user,
            self.r_ul_user,
            self.phi_dl_user,
            self.phi_ul_user,
            self.theta_dl_user,
            self.theta_ul_user,
        ) = self._sample_users()
        self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus = self._sample_focus_for_user(
            self.r_dl_user, self.phi_dl_user, self.theta_dl_user, reset_mode
        )
        self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus = self._sample_focus_for_user(
            self.r_ul_user, self.phi_ul_user, self.theta_ul_user, reset_mode
        )
        self.current_SI_dB = self.compute_SI_dB(
            self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus, self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus
        )
        self.prev_SI_dB = self.current_SI_dB
        self.best_SI_dB = self.current_SI_dB
        return self._get_obs(), {}

    def reset_fixed(
        self,
        phi_dl_user=20.0,
        phi_ul_user=30.0,
        theta_dl_user=0.0,
        theta_ul_user=0.0,
        r_dl_user=2.0,
        r_ul_user=2.0,
        seed=None,
        reset_mode="near_user",
        options=None,
    ):
        if seed is not None:
            self._rng.seed(seed)
        if options is not None and "reset_mode" in options:
            reset_mode = options["reset_mode"]
        self.step_idx = 0
        self.phi_dl_user = float(phi_dl_user)
        self.phi_ul_user = float(phi_ul_user)
        self.theta_dl_user = float(theta_dl_user)
        self.theta_ul_user = float(theta_ul_user)
        self.r_dl_user = float(r_dl_user)
        self.r_ul_user = float(r_ul_user)
        self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus = self._sample_focus_for_user(
            self.r_dl_user, self.phi_dl_user, self.theta_dl_user, reset_mode
        )
        self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus = self._sample_focus_for_user(
            self.r_ul_user, self.phi_ul_user, self.theta_ul_user, reset_mode
        )
        self.current_SI_dB = self.compute_SI_dB(
            self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus, self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus
        )
        self.prev_SI_dB = self.current_SI_dB
        self.best_SI_dB = self.current_SI_dB
        return self._get_obs(), {}

    def step(self, action):
        self.step_idx += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if self.direct_action_mode:
            self.r_tx_focus = self._action_to_range(action[0])
            self.phi_tx_focus = self._action_to_angle(action[1], self.phi_min, self.phi_max)
            self.theta_tx_focus = self._action_to_angle(action[2], self.theta_min, self.theta_max)
            self.r_rx_focus = self._action_to_range(action[3])
            self.phi_rx_focus = self._action_to_angle(action[4], self.phi_min, self.phi_max)
            self.theta_rx_focus = self._action_to_angle(action[5], self.theta_min, self.theta_max)
        else:
            self.r_tx_focus += self.max_delta_r * action[0]
            self.phi_tx_focus += self.max_delta_phi * action[1]
            self.theta_tx_focus += self.max_delta_theta * action[2]
            self.r_rx_focus += self.max_delta_r * action[3]
            self.phi_rx_focus += self.max_delta_phi * action[4]
            self.theta_rx_focus += self.max_delta_theta * action[5]

        self.r_tx_focus = np.clip(self.r_tx_focus, self.r_min, self.r_max)
        self.r_rx_focus = np.clip(self.r_rx_focus, self.r_min, self.r_max)
        self.phi_tx_focus = np.clip(self.phi_tx_focus, self.phi_min, self.phi_max)
        self.phi_rx_focus = np.clip(self.phi_rx_focus, self.phi_min, self.phi_max)
        self.theta_tx_focus = np.clip(self.theta_tx_focus, self.theta_min, self.theta_max)
        self.theta_rx_focus = np.clip(self.theta_rx_focus, self.theta_min, self.theta_max)

        prev_si = self.current_SI_dB
        prev_best_si = self.best_SI_dB
        si_now = self.compute_SI_dB(
            self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus, self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus
        )
        penalty = self.compute_focus_penalty()
        boundary_penalty = self.compute_boundary_penalty()
        gain_metrics = self.compute_gain_metrics()
        si_ref = self.compute_SI_reference_dB()
        action_magnitude = float(np.mean(np.square(action)))
        terminated = False
        truncated = self.step_idx >= self.max_steps

        reward = self.compute_reward(
            si_now,
            prev_si,
            prev_best_si,
            si_ref,
            penalty,
            boundary_penalty,
            action_magnitude,
            gain_metrics,
            truncated,
        )
        self.prev_SI_dB = prev_si
        self.current_SI_dB = si_now
        if self.reward_mode == "neg_si_gain_constraint":
            dl_gain_feasible = gain_metrics["DL_gain_loss_dB"] <= self.dl_gain_margin_db
            ul_gain_feasible = gain_metrics["UL_gain_loss_dB"] <= self.ul_gain_margin_db
            self.best_SI_dB = min(prev_best_si, si_now) if dl_gain_feasible and ul_gain_feasible else prev_best_si
        else:
            self.best_SI_dB = min(prev_best_si, si_now)

        info = {
            "SI_dB": float(si_now),
            "prev_SI_dB": float(prev_si),
            "delta_SI_dB": float(si_now - prev_si),
            "best_SI_dB": float(self.best_SI_dB),
            "regret_SI_dB": float(si_now - self.best_SI_dB),
            "SI_ref_dB": float(si_ref),
            "focus_penalty": float(penalty),
            "boundary_penalty": float(boundary_penalty),
            "action_magnitude": float(action_magnitude),
            "DL_gain_ref_dB": gain_metrics["DL_gain_ref_dB"],
            "DL_gain_dB": gain_metrics["DL_gain_dB"],
            "DL_gain_loss_dB": gain_metrics["DL_gain_loss_dB"],
            "DL_gain_margin_dB": float(self.dl_gain_margin_db),
            "UL_gain_ref_dB": gain_metrics["UL_gain_ref_dB"],
            "UL_gain_dB": gain_metrics["UL_gain_dB"],
            "UL_gain_loss_dB": gain_metrics["UL_gain_loss_dB"],
            "UL_gain_margin_dB": float(self.ul_gain_margin_db),
            "reward": float(reward),
            "r_tx_focus": float(self.r_tx_focus),
            "phi_tx_focus": float(self.phi_tx_focus),
            "theta_tx_focus": float(self.theta_tx_focus),
            "r_rx_focus": float(self.r_rx_focus),
            "phi_rx_focus": float(self.phi_rx_focus),
            "theta_rx_focus": float(self.theta_rx_focus),
            "r_dl_user": float(self.r_dl_user),
            "phi_dl_user": float(self.phi_dl_user),
            "theta_dl_user": float(self.theta_dl_user),
            "r_ul_user": float(self.r_ul_user),
            "phi_ul_user": float(self.phi_ul_user),
            "theta_ul_user": float(self.theta_ul_user),
            "tx_ports_1_based": (self.tx_idx + 1).tolist(),
            "rx_ports_1_based": (self.rx_idx + 1).tolist(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        si = self.compute_SI_dB(
            self.r_tx_focus, self.phi_tx_focus, self.theta_tx_focus, self.r_rx_focus, self.phi_rx_focus, self.theta_rx_focus
        )
        print(
            f"step {self.step_idx:2d} | "
            f"Tx focus: r={self.r_tx_focus:.3f} m, phi={self.phi_tx_focus:.2f} deg, theta={self.theta_tx_focus:.2f} deg | "
            f"Rx focus: r={self.r_rx_focus:.3f} m, phi={self.phi_rx_focus:.2f} deg, theta={self.theta_rx_focus:.2f} deg | "
            f"DL user: r={self.r_dl_user:.3f} m, phi={self.phi_dl_user:.2f} deg, theta={self.theta_dl_user:.2f} deg | "
            f"UL user: r={self.r_ul_user:.3f} m, phi={self.phi_ul_user:.2f} deg, theta={self.theta_ul_user:.2f} deg | "
            f"SI={si:.2f} dB"
        )
