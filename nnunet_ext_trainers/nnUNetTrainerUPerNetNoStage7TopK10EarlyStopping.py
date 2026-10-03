"""Dataset501 eight-stage UPerNet ablation: physically remove encoder stage7."""

from torch import nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from nnUNetTrainerMixins import _PlainConvUNetUPerNet
from nnUNetTrainerUPerNetTopK10EarlyStopping import nnUNetTrainerUPerNetTopK10EarlyStopping


class nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping(nnUNetTrainerUPerNetTopK10EarlyStopping):
    """Keep the baseline Trainer policy and stages0..6; use features (1,3,5,6)."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if enable_deep_supervision:
            raise ValueError("NoStage7 UPerNet does not support deep supervision")
        if tuple(configuration_manager.patch_size) != (512, 512):
            raise ValueError("NoStage7 UPerNet requires the Dataset501 2D 512 x 512 configuration")
        if configuration_manager.network_arch_class_name != (
            "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        ):
            raise ValueError("NoStage7 UPerNet requires the official PlainConvUNet architecture")

        kwargs = configuration_manager.network_arch_init_kwargs
        if kwargs.get("deep_supervision", False):
            raise ValueError("NoStage7 UPerNet does not support deep supervision in architecture kwargs")
        if kwargs.get("n_stages") != 8:
            raise ValueError("NoStage7 UPerNet requires an original encoder with eight stages")
        for name, count in (
            ("features_per_stage", 8),
            ("kernel_sizes", 8),
            ("strides", 8),
            ("n_conv_per_stage", 8),
            ("n_conv_per_stage_decoder", 7),
        ):
            values = kwargs.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != count:
                raise ValueError(f"NoStage7 UPerNet requires {count} explicit {name} entries")
        expected_strides = ((1, 1),) + ((2, 2),) * 7
        if any(
            not isinstance(stride, (list, tuple)) or tuple(stride) != expected
            for stride, expected in zip(kwargs["strides"], expected_strides)
        ):
            raise ValueError("NoStage7 UPerNet requires the Dataset501 eight-stage 2D stride geometry")

        # The official helper copies the kwargs. Build the original encoder so
        # all retained layers have exactly their supplied plan configuration.
        official_network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=False,
        )
        if not isinstance(official_network, PlainConvUNet):
            raise ValueError("NoStage7 UPerNet requires the official PlainConvUNet architecture")
        encoder = official_network.encoder
        if encoder.conv_op is not nn.Conv2d or len(encoder.stages) != 8:
            raise ValueError("NoStage7 UPerNet requires an eight-stage Conv2d encoder")

        # PlainConvEncoder forward/accounting iterate len(stages). Remove the
        # module and trim its parallel metadata, without touching the plans.
        encoder.stages = encoder.stages[:7]
        encoder.output_channels = encoder.output_channels[:7]
        encoder.strides = encoder.strides[:7]
        encoder.kernel_sizes = encoder.kernel_sizes[:7]
        model = _PlainConvUNetUPerNet(
            encoder,
            (1, 3, 5, 6),
            num_output_channels,
            norm_op=encoder.norm_op,
            norm_op_kwargs=encoder.norm_op_kwargs,
            conv_bias=encoder.conv_bias,
        )
        model.decoder.apply(PlainConvUNet.initialize)
        return model
