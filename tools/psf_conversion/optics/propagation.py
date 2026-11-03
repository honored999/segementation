import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union

class AngularSpectrumPropagation(nn.Module):
    def __init__(self, distance: float, wavelength: float, pixel_size: float, 
                 device: str = 'cpu', epsilon: float = 1e-10, padding: int = 200):
        """
        角谱传播法初始化
        
        参数:
            distance: 传播距离(m)
            wavelength: 波长(m)
            pixel_size: x和y方向的采样间隔(m)
            device: 计算设备
            epsilon: 数值稳定性的小常数
            padding: 每边添加的padding像素数，用于减少边界效应
        """
        super(AngularSpectrumPropagation, self).__init__()
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.device = device
        self.epsilon = epsilon
        self.padding = padding
        self.transfer_function = None
        self.cached_shape = None

    def precompute_transfer_function(self, height, width):
        """
        预计算传递函数以提高效率
        """
        # 清理旧的传递函数以节省内存
        if self.transfer_function is not None:
            del self.transfer_function
            torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        # 创建频率网格 (单位: 1/m)
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, self.pixel_size)).to(self.device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, self.pixel_size)).to(self.device)
        
        # 兼容不同版本的PyTorch
        try:
            FX, FY = torch.meshgrid(fy, fx, indexing='ij')
        except TypeError:
            # 对于不支持indexing参数的旧版PyTorch
            FX, FY = torch.meshgrid(fy, fx)
            # 确保维度顺序正确
            if FX.shape != (height, width):
                FX = FX.transpose(0, 1)
                FY = FY.transpose(0, 1)
        
        # 计算归一化频率平方和: (λ*fx)² + (λ*fy)²
        # 这决定了传播波 vs 消逝波的分界
        normalized_freq_squared = (self.wavelength * FX)**2 + (self.wavelength * FY)**2
        
        # 传播区域内的频率（归一化频率平方和 < 1）
        propagating_mask = normalized_freq_squared < 1.0
        
        # 初始化传递函数
        self.transfer_function = torch.zeros(height, width, dtype=torch.cfloat, device=self.device)
        
        # 对于传播区域：计算z方向波矢量分量
        # k_z = (2π/λ) * sqrt(1 - (λ*fx)² - (λ*fy)²)
        sqrt_term = torch.sqrt(1 - normalized_freq_squared[propagating_mask] + self.epsilon)
        k_z = 2 * torch.pi * sqrt_term / self.wavelength
        phase = k_z * self.distance
        self.transfer_function[propagating_mask] = torch.exp(1j * phase)
        
        # 对于消逝波（非传播区域）：使用指数衰减
        if not propagating_mask.all():
            # 消逝波衰减系数: exp(-|k_z| * z)，其中 |k_z| = (2π/λ) * sqrt((λ*fx)² + (λ*fy)² - 1)
            decay_coeff = 2 * torch.pi * torch.sqrt(normalized_freq_squared[~propagating_mask] - 1 + self.epsilon) / self.wavelength
            evanescent_decay = torch.exp(-decay_coeff * self.distance)
             # 确保数据类型匹配：将实数衰减转换为复数
            self.transfer_function[~propagating_mask] = evanescent_decay.to(torch.cfloat)
            
        # 缓存形状以便检测变化
        self.cached_shape = (height, width)

    def forward(self, complex_field):
        """
        前向传播函数（带padding以减少边界效应）
        
        参数:
            complex_field: 复振幅场，shape为[batch_size, channels, height, width]或[batch_size, height, width]，dtype为torch.cfloat
        
        返回:
            传播后的复振幅场，形状与输入相同，dtype为torch.cfloat
        """
        # 获取原始场的尺寸
        original_shape = complex_field.shape
        if len(original_shape) == 4:  # [B, C, H, W]
            batch_size, channels, height, width = original_shape
            field_2d = complex_field.view(-1, height, width)  # [B*C, H, W]
        else:  # [B, H, W]
            batch_size, height, width = original_shape
            channels = 1
            field_2d = complex_field
        
        # 添加padding
        if self.padding > 0:
            padded_field = torch.nn.functional.pad(
                field_2d, 
                (self.padding, self.padding, self.padding, self.padding), 
                mode='constant', 
                value=0
            )
            padded_height = height + 2 * self.padding
            padded_width = width + 2 * self.padding
        else:
            padded_field = field_2d
            padded_height, padded_width = height, width
        
        # 检查形状是否变化，如变化或传递函数未初始化则重新计算
        if self.transfer_function is None or self.cached_shape != (padded_height, padded_width):
            self.precompute_transfer_function(padded_height, padded_width)

        # 执行FFT (使用批处理支持的方式)
        field_fft = torch.fft.fft2(padded_field)
        field_fft = torch.fft.fftshift(field_fft, dim=(-2, -1))
        
        # 应用传递函数 (确保广播正确)
        field_fft = field_fft * self.transfer_function
        
        # 执行IFFT
        field_fft = torch.fft.ifftshift(field_fft, dim=(-2, -1))
        field_propagated = torch.fft.ifft2(field_fft)
        
        # 移除padding，恢复原始尺寸
        if self.padding > 0:
            field_propagated = field_propagated[
                ..., 
                self.padding:self.padding + height, 
                self.padding:self.padding + width
            ]
        
        # 恢复原始形状
        if len(original_shape) == 4:
            field_propagated = field_propagated.view(batch_size, channels, height, width)
        
        return field_propagated
        
    def __del__(self):
        """析构函数，清理GPU内存"""
        if hasattr(self, 'transfer_function') and self.transfer_function is not None:
            del self.transfer_function
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


class MultiWavelengthAngularSpectrumPropagation(nn.Module):
    """
    支持多波长的角谱传播法，用于RGB色散光学系统
    """
    
    def __init__(self, distance: float, wavelengths: Dict[str, float], 
                 pixel_size: float, device: str = 'cpu', epsilon: float = 1e-10, padding: int = 200):
        """
        多波长角谱传播法
        
        参数:
            distance: 传播距离(m)
            wavelengths: 波长字典 {'red': 650e-9, 'green': 532e-9, 'blue': 450e-9}
            pixel_size: x和y方向的采样间隔(m)
            device: 计算设备
            epsilon: 数值稳定性的小常数
            padding: 每边添加的padding像素数，用于减少边界效应
        """
        super(MultiWavelengthAngularSpectrumPropagation, self).__init__()
        self.distance = distance
        self.wavelengths = wavelengths
        self.pixel_size = pixel_size
        self.device = device
        self.epsilon = epsilon
        self.padding = padding
        
        # 为每个波长创建单独的传播算子
        self.propagators = nn.ModuleDict()
        for color, wavelength in wavelengths.items():
            self.propagators[color] = AngularSpectrumPropagation(
                distance=distance,
                wavelength=wavelength,
                pixel_size=pixel_size,
                device=device,
                epsilon=epsilon,
                padding=padding
            )
        
        # RGB通道顺序
        self.color_order = ['red', 'green', 'blue']
    
    def forward(self, complex_fields_rgb: torch.Tensor) -> torch.Tensor:
        """
        多波长前向传播
        
        参数:
            complex_fields_rgb: RGB复振幅场，shape为[batch_size, 3, height, width]，dtype为torch.cfloat
        
        返回:
            传播后的RGB复振幅场，形状与输入相同，dtype为torch.cfloat
        """
        # 输入验证
        if complex_fields_rgb.dim() != 4:
            raise ValueError(f"期望4D输入张量 [batch, channels, height, width]，但得到{complex_fields_rgb.dim()}D")
        
        batch_size, channels, height, width = complex_fields_rgb.shape
        
        # 检查通道数
        if channels != 3:
            raise ValueError(f"多波长传播期望3个RGB通道，但得到{channels}个通道")
        
        # 确保输入在正确的设备上
        if complex_fields_rgb.device != torch.device(self.device):
            complex_fields_rgb = complex_fields_rgb.to(self.device)
        
        # 确保输入是复数类型
        if not complex_fields_rgb.dtype.is_complex:
            raise ValueError(f"期望复数输入，但得到{complex_fields_rgb.dtype}")
        
        # 分别对每个波长进行传播
        output_fields = []
        
        for i, color in enumerate(self.color_order):
            # 获取对应颜色通道的复场
            field = complex_fields_rgb[:, i]  # [batch_size, height, width]
            
            # 使用对应波长的传播算子
            propagated_field = self.propagators[color](field)
            
            output_fields.append(propagated_field)
        
        # 堆叠回RGB格式
        return torch.stack(output_fields, dim=1)
    
    def precompute_transfer_functions(self, height: int, width: int):
        """
        为所有波长预计算传递函数
        """
        for propagator in self.propagators.values():
            propagator.precompute_transfer_function(height, width)
    
    def get_wavelength_info(self) -> Dict[str, float]:
        """
        获取波长信息
        """
        return self.wavelengths.copy()
    
    def __del__(self):
        """析构函数，清理GPU内存"""
        if hasattr(self, 'propagators'):
            for propagator in self.propagators.values():
                if hasattr(propagator, '__del__'):
                    propagator.__del__()


def create_propagation_layer(distance: float, 
                           wavelength: Union[float, Dict[str, float]], 
                           pixel_size: float, 
                           device: str = 'cuda', 
                           epsilon: float = 1e-10,
                           padding: int = 200) -> nn.Module:
    """
    工厂函数：根据波长参数创建合适的传播层
    
    参数:
        distance: 传播距离(m)
        wavelength: 单波长(float)或多波长字典(Dict[str, float])
        pixel_size: 像素尺寸(m)
        device: 计算设备
        epsilon: 数值稳定性常数
        padding: 每边添加的padding像素数，用于减少边界效应
    
    返回:
        AngularSpectrumPropagation 或 MultiWavelengthAngularSpectrumPropagation
    """
    if isinstance(wavelength, dict):
        # 多波长模式
        return MultiWavelengthAngularSpectrumPropagation(
            distance=distance,
            wavelengths=wavelength,
            pixel_size=pixel_size,
            device=device,
            epsilon=epsilon,
            padding=padding
        )
    else:
        # 单波长模式
        return AngularSpectrumPropagation(
            distance=distance,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=device,
            epsilon=epsilon,
            padding=padding
        ) 
