"""拟合的相位-宽度函数（基于 PhaseLibrary_P400_H600_W80-360.npz）。

周期 400nm，高度 600nm，宽度范围 80-320nm。
"""

import torch
import numpy as np
from typing import Union


# 多项式系数（从高次到低次）
# phase = c[0]*w^n + c[1]*w^(n-1) + ... + c[n]
PHASE_COEFFS = {
    405: np.array([3.608536817324416e-18, -6.420158559417363e-15, 4.959298501657753e-12, -2.178518274411385e-09, 5.982606896034627e-07, -1.061915775976196e-04, 1.213944138290035e-02, -8.585664870031612e-01, 3.403519381661430e+01, -5.767856928822478e+02]),
    450: np.array([1.615869411804115e-18, -2.988762383158265e-15, 2.410596606173123e-12, -1.111527820010239e-09, 3.224279198112519e-07, -6.088621422864561e-05, 7.461008558943823e-03, -5.696800531631298e-01, 2.452033214729905e+01, -4.528058504256674e+02]),
    520: np.array([-7.258928777178854e-19, 1.348582593276442e-15, -1.090885236189239e-12, 5.027147408756554e-10, -1.449241644864077e-07, 2.700255281585115e-05, -3.241688849372641e-03, 2.415215420764799e-01, -1.012568729219527e+01, 1.818789302891334e+02]),
    532: np.array([-3.408842696061169e-19, 6.336426937147136e-16, -5.134805750800076e-13, 2.372485137668206e-10, -6.858099449437712e-08, 1.280266445157907e-05, -1.537723694588967e-03, 1.145292012420696e-01, -4.795744772596711e+00, 8.589147049866438e+01]),
    635: np.array([2.331529203304141e-19, -4.115619370061918e-16, 3.141738395487293e-13, -1.358222543712273e-10, 3.658278599399100e-08, -6.361435320288894e-06, 7.143969932447666e-04, -4.993210924769290e-02, 1.975487110973436e+00, -3.388388093910542e+01]),
}

# 拟合有效宽度范围
MAX_WIDTH_NM = 320.0


def compute_phase_fitted(
    width_nm: Union[torch.Tensor, np.ndarray, float],
    wavelength_nm: int,
) -> Union[torch.Tensor, np.ndarray, float]:
    """根据柱宽度计算相位（使用多项式拟合）。
    
    Args:
        width_nm: 柱宽度（nm），范围 [80, 360]
        wavelength_nm: 波长（nm），支持 405, 450, 520, 532, 635
    
    Returns:
        相位（rad）
    """
    if wavelength_nm not in PHASE_COEFFS:
        raise ValueError(f"Unsupported wavelength: {wavelength_nm}. "
                        f"Supported: {list(PHASE_COEFFS.keys())}")
    
    coeffs = PHASE_COEFFS[wavelength_nm]
    
    if isinstance(width_nm, torch.Tensor):
        # PyTorch 版本
        result = torch.zeros_like(width_nm)
        for i, c in enumerate(coeffs):
            power = len(coeffs) - 1 - i
            result = result + c * (width_nm ** power)
        return result
    else:
        # NumPy 版本
        return np.polyval(coeffs, width_nm)


def compute_phase_all_wavelengths(
    width_nm: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """计算所有三个波长的相位。
    
    Args:
        width_nm: 柱宽度（nm）
    
    Returns:
        相位数组，形状为 (3, *width_nm.shape)，顺序为 [B, G, R]
    """
    phases = []
    for wl in [450, 520, 635]:
        phases.append(compute_phase_fitted(width_nm, wl))
    
    if isinstance(width_nm, torch.Tensor):
        return torch.stack(phases, dim=0)
    else:
        return np.stack(phases, axis=0)
