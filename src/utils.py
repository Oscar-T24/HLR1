"""Utility functions for HFSS loading, near-field channels, and beamforming."""

from __future__ import annotations

import numpy as np
import skrf as rf


def load_hfss_channel(hfss_file, target_freq=3.5e9):
    ntwk = rf.Network(str(hfss_file))
    freq_idx = np.argmin(np.abs(ntwk.f - target_freq))
    hsi = ntwk.s[freq_idx, :, :].astype(np.complex128)
    return hsi, float(ntwk.f[freq_idx])


def make_ula_positions(num_elements: int, element_spacing: float, axis="y") -> np.ndarray:
    coord = (np.arange(num_elements) - (num_elements - 1) / 2) * element_spacing
    x = np.zeros(num_elements)
    y = np.zeros(num_elements)
    z = np.zeros(num_elements)
    if axis == "x":
        x = coord
    elif axis == "y":
        y = coord
    elif axis == "z":
        z = coord
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return np.stack([x, y, z], axis=1)


def sph_to_cart(r: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z])


def near_field_channel_to_user(pos_elem: np.ndarray, user_sph: np.ndarray, fc: float, c: float = 3e8) -> np.ndarray:
    wavelength = c / fc
    k = 2 * np.pi / wavelength
    r0 = user_sph[0]
    theta_deg = user_sph[1]
    phi_deg = user_sph[2]
    user_cart = sph_to_cart(r0, theta_deg, phi_deg)
    diff = pos_elem - user_cart
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return (r0 / dist) * np.exp(-1j * k * (dist - r0))


def conjugate_beamforming(h: np.ndarray) -> np.ndarray:
    norm_h = np.linalg.norm(h)
    if norm_h < 1e-12:
        raise ValueError("Channel norm is too small. Check user position or array geometry.")
    return np.conj(h) / norm_h


def fd_beamforming(pos_i: np.ndarray, pos_j: np.ndarray, user_i_sph: np.ndarray, user_j_sph: np.ndarray, fc: float):
    h_i_ui = near_field_channel_to_user(pos_i, user_i_sph, fc)
    h_j_uj = near_field_channel_to_user(pos_j, user_j_sph, fc)
    return conjugate_beamforming(h_i_ui), conjugate_beamforming(h_j_uj)


def near_field_conjugate_weights(pos_elem, r, phi_deg, theta_deg=0.0, fc=3.5e9):
    user_sph = np.array([r, theta_deg, phi_deg], dtype=np.float64)
    h = near_field_channel_to_user(pos_elem, user_sph, fc)
    return conjugate_beamforming(h)


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return None
    kernel = np.ones(window, dtype=np.float64)
    valid = np.isfinite(values).astype(np.float64)
    sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="valid")
    counts = np.convolve(valid, kernel, mode="valid")
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0.0)
