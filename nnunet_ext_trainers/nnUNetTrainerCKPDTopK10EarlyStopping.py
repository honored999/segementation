"""Opt-in Dataset501 CKPD adaptation of the existing TopK10/early-stop Trainer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Union

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results

import torch
from torch._dynamo import OptimizedModule
from torch.nn.utils import parametrize

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerTopK10EarlyStopping import nnUNetTrainerTopK10EarlyStopping

from ckpd import (
    CKPD_SCHEMA_VERSION,
    CKPDWeightParametrization,
    Conv2dDecomposition,
    apply_ckpd_adapters,
    atomic_torch_save,
    capture_rng_state,
    encoder_conv_layers,
    encoder_semantic_signature,
    hash_file,
    hash_identifier,
    merged_state_dict,
    module_fingerprint,
    restore_rng_state,
    validate_rng_state,
)

_restore_rng_state_direct = restore_rng_state


def _read_fold(split_path: Path, fold: int) -> tuple[str, set[str], set[str]]:
    if not split_path.is_file():
        raise RuntimeError(f"fixed nnU-Net split file is required and will not be generated: {split_path}")
    digest = hash_file(split_path)
    try:
        splits = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read fixed split file {split_path}: {exc}") from exc
    if not isinstance(splits, list) or fold < 0 or fold >= len(splits):
        raise RuntimeError(f"requested fold {fold} is absent from {split_path}")
    entry = splits[fold]
    if not isinstance(entry, dict) or not isinstance(entry.get("train"), list) or not isinstance(entry.get("val"), list):
        raise RuntimeError(f"fold {fold} in {split_path} must contain train and val case lists")
    train = {str(identifier) for identifier in entry["train"]}
    val = {str(identifier) for identifier in entry["val"]}
    if not train or not val or train.intersection(val):
        raise RuntimeError(f"fold {fold} in {split_path} has empty or overlapping train/validation IDs")
    return digest, train, val


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_results_root_isolated() -> Path:
    results_root = Path(nnUNet_results.require()).resolve()
    for variable, protected in (("nnUNet_raw", nnUNet_raw), ("nnUNet_preprocessed", nnUNet_preprocessed)):
        if protected.is_set():
            protected_root = Path(os.fspath(protected)).resolve()
            if _path_is_within(results_root, protected_root) or _path_is_within(protected_root, results_root):
                raise RuntimeError(f"nnUNet_results overlaps protected {variable}: {results_root} vs {protected_root}")
    return results_root


def _has_prior_run_artifacts(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(next(folder.glob(pattern), None) is not None
               for pattern in ("checkpoint_*.pth", "training_log_*.txt", "debug.json"))


def _module_aliases(network: torch.nn.Module, target: torch.nn.Module) -> list[str]:
    return [name for name, module in network.named_modules(remove_duplicate=False) if module is target]


class nnUNetTrainerCKPDTopK10EarlyStopping(nnUNetTrainerTopK10EarlyStopping):
    """CKPD encoder adapters, trainable decoder, and inherited TopK10/ES behavior.

    A fresh run requires ``NNUNET_CKPD_CALIBRATION`` and
    ``NNUNET_CKPD_SOURCE_CHECKPOINT``. Adapters are installed in run_training,
    after nnU-Net's official pretrained-weight loading step. Same-run resume
    reconstructs adapters from its checkpoint and does not read those files.
    """

    CKPD_DEFAULT_RANK = 4
    CKPD_DEFAULT_LAYER_COUNT = 4
    CKPD_TRAINING_STATE_KEY = "ckpd_training_state"

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")) -> None:
        # Check output placement and snapshot old-run evidence before nnU-Net
        # creates this invocation's output folder and training log. Validation-only
        # instances remain allowed to use an existing run folder.
        results_root = _assert_results_root_isolated()
        dataset_name = plans.get("dataset_name")
        plans_name = plans.get("plans_name")
        if not isinstance(dataset_name, str) or not isinstance(plans_name, str):
            raise RuntimeError("nnU-Net plans must identify the dataset and plans before CKPD startup")
        run_name = f"{self.__class__.__name__}__{plans_name}__{configuration}"
        run_folder = results_root / dataset_name / run_name / f"fold_{fold}"
        self._ckpd_prior_run_artifacts = _has_prior_run_artifacts(run_folder)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._ckpd_adapters_installed = False
        self._ckpd_resumed = False
        self._ckpd_selected_layers: list[str] = []
        self._ckpd_artifact_sha256: str | None = None
        self._ckpd_source_sha256: str | None = None
        self._ckpd_encoder_fingerprint: str | None = None
        self._ckpd_target_split_sha256: str | None = None
        self._ckpd_target_plans_sha256: str | None = None

    def configure_optimizers(self):
        trainable = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("CKPD Trainer has no trainable parameters")
        optimizer = torch.optim.SGD(
            trainable,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler

    def _target_split_identity(self) -> tuple[str, set[str], set[str]]:
        if int(self.fold) != 0:
            raise RuntimeError("this approved CKPD run is restricted to Dataset501 fold 0")
        if str(self.configuration_name) != "2d":
            raise RuntimeError("this approved CKPD run supports only the ordinary 2d configuration")
        dataset_dir = Path(self.preprocessed_dataset_folder_base)
        if dataset_dir.name != "Dataset501_StrokeLesion":
            raise RuntimeError(
                "this CKPD Trainer is scoped to Dataset501_StrokeLesion; "
                f"resolved preprocessed dataset is {dataset_dir}"
            )
        return _read_fold(dataset_dir / "splits_final.json", int(self.fold))

    def _target_plans_identity(self) -> str:
        plans = getattr(self.plans_manager, "plans", None)
        if not isinstance(plans, dict):
            raise RuntimeError("nnU-Net plans manager does not expose its resolved plans dictionary")
        payload = json.dumps(plans, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _assert_target_split_unchanged(self) -> str:
        digest, _, _ = self._target_split_identity()
        plan_digest = self._target_plans_identity()
        if self._ckpd_target_split_sha256 is None:
            self._ckpd_target_split_sha256 = digest
        elif self._ckpd_target_split_sha256 != digest:
            raise RuntimeError("Dataset501 splits_final.json changed during this CKPD run")
        if self._ckpd_target_plans_sha256 is None:
            self._ckpd_target_plans_sha256 = plan_digest
        elif self._ckpd_target_plans_sha256 != plan_digest:
            raise RuntimeError("Dataset501 nnU-Net plans changed during this CKPD run")
        return digest

    def get_dataloaders(self):
        # Guard before nnU-Net can call do_split or unpack prepared data.
        self._assert_target_split_unchanged()
        return super().get_dataloaders()

    @staticmethod
    def _require_supported_network(network: torch.nn.Module, is_ddp: bool, compile_enabled: bool) -> None:
        if is_ddp:
            raise RuntimeError("CKPD Trainer currently rejects DDP; use one process/device")
        if compile_enabled or isinstance(network, OptimizedModule):
            raise RuntimeError("CKPD Trainer currently rejects torch.compile")
        if hasattr(network, "module") and isinstance(network.module, torch.nn.Module):
            raise RuntimeError("CKPD Trainer requires the unwrapped PlainConvUNet network")
        if type(network).__name__ != "PlainConvUNet" or not hasattr(network, "encoder") or not hasattr(network, "decoder"):
            raise RuntimeError("CKPD Trainer supports only the ordinary 2D PlainConvUNet architecture")
        if not encoder_conv_layers(network):
            raise RuntimeError("CKPD PlainConvUNet has no supported encoder Conv2d layers")

    def _load_and_validate_calibration(self) -> tuple[dict, str, str]:
        artifact_path = os.environ.get("NNUNET_CKPD_CALIBRATION")
        source_checkpoint = os.environ.get("NNUNET_CKPD_SOURCE_CHECKPOINT")
        if not artifact_path or not source_checkpoint:
            raise RuntimeError(
                "fresh CKPD training requires NNUNET_CKPD_CALIBRATION and "
                "NNUNET_CKPD_SOURCE_CHECKPOINT; no full-tuning fallback is permitted"
            )
        artifact_file = Path(artifact_path)
        source_file = Path(source_checkpoint)
        if not artifact_file.is_file() or not source_file.is_file():
            raise RuntimeError("CKPD calibration or source pretrained checkpoint path does not exist")
        artifact_sha = hash_file(artifact_file)
        source_sha = hash_file(source_file)
        try:
            artifact = torch.load(artifact_file, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"cannot load CKPD calibration artifact: {exc}") from exc
        required = {
            "schema_version", "source_dataset", "source_fold", "source_split_path",
            "source_split_sha256", "source_train_case_hashes", "source_val_case_hashes",
            "source_encoder_semantic_signature", "patch_size", "selected_case_hashes",
            "source_checkpoint_sha256", "source_encoder_fingerprint", "configuration",
            "source_trainer_name", "source_plans_name", "source_data_identifier", "source_dataset_name",
            "source_plans_path", "source_plans_sha256", "source_dataset_json_path",
            "source_dataset_json_sha256", "encoder_conv_metadata", "selected_layers",
            "sample_count", "seed", "covariance", "max_positions_per_case_per_layer",
            "damping_relative", "max_covariance_mib", "asr_definition", "rank", "layer_count", "factors",
        }
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise RuntimeError("CKPD calibration artifact is missing required metadata")
        if artifact["schema_version"] != CKPD_SCHEMA_VERSION:
            raise RuntimeError("unsupported CKPD calibration schema")
        if artifact["source_dataset"] != "Dataset508_ISLES2022DWI" or artifact["source_fold"] != 0:
            raise RuntimeError("calibration must bind Dataset508_ISLES2022DWI fold 0")
        if artifact["configuration"] != "2d":
            raise RuntimeError("calibration artifact is not for the 2d configuration")
        if (artifact["source_trainer_name"] != "nnUNetTrainerTopK10EarlyStopping"
                or artifact["source_plans_name"] != "nnUNetPlansFrom501"
                or artifact["source_dataset_name"] != "Dataset508_ISLES2022DWI"
                or not isinstance(artifact["source_data_identifier"], str)
                or not artifact["source_data_identifier"]):
            raise RuntimeError("calibration source trainer/plans/dataset identity is unsupported")
        sample_count = artifact["sample_count"]
        if (not isinstance(sample_count, int) or sample_count < 1
                or sample_count != len(artifact["selected_case_hashes"])):
            raise RuntimeError("calibration sample_count does not match selected source-train cases")
        if (not isinstance(artifact["seed"], int) or not isinstance(artifact["covariance"], str)
                or not isinstance(artifact["asr_definition"], str)
                or not artifact["asr_definition"].startswith("descending SVD(W C_reg): sigma[-rank] / sigma[-1]")
                or not isinstance(artifact["max_positions_per_case_per_layer"], int)
                or artifact["max_positions_per_case_per_layer"] < 1
                or not isinstance(artifact["max_covariance_mib"], int)
                or artifact["max_covariance_mib"] < 1
                or not isinstance(artifact["damping_relative"], (int, float))
                or not torch.isfinite(torch.tensor(float(artifact["damping_relative"])))
                or float(artifact["damping_relative"]) < 0):
            raise RuntimeError("calibration sampling, covariance, damping, or ASR metadata is invalid")
        if artifact["source_checkpoint_sha256"] != source_sha:
            raise RuntimeError("pretrained source checkpoint SHA256 differs from calibration")

        source_split = Path(artifact["source_split_path"])
        source_plans = Path(artifact["source_plans_path"])
        source_dataset_json = Path(artifact["source_dataset_json_path"])
        split_hash, source_train, source_val = _read_fold(source_split, int(artifact["source_fold"]))
        if split_hash != artifact["source_split_sha256"]:
            raise RuntimeError("Dataset508 splits_final.json differs from calibration")
        if (not source_plans.is_file() or hash_file(source_plans) != artifact["source_plans_sha256"]
                or not source_dataset_json.is_file()
                or hash_file(source_dataset_json) != artifact["source_dataset_json_sha256"]):
            raise RuntimeError("Dataset508 plans or dataset.json differs from calibration")
        expected_source_train_hashes = {hash_identifier(identifier) for identifier in source_train}
        expected_source_val_hashes = {hash_identifier(identifier) for identifier in source_val}
        stored_train_hashes = artifact["source_train_case_hashes"]
        stored_val_hashes = artifact["source_val_case_hashes"]
        if (not isinstance(stored_train_hashes, list) or len(stored_train_hashes) != len(expected_source_train_hashes)
                or set(stored_train_hashes) != expected_source_train_hashes):
            raise RuntimeError("calibration source-train manifest does not match Dataset508 fold 0")
        if (not isinstance(stored_val_hashes, list) or len(stored_val_hashes) != len(expected_source_val_hashes)
                or set(stored_val_hashes) != expected_source_val_hashes):
            raise RuntimeError("calibration source-validation manifest does not match Dataset508 fold 0")
        selected_hashes = set(artifact["selected_case_hashes"])
        if not selected_hashes or not selected_hashes.issubset(expected_source_train_hashes):
            raise RuntimeError("calibration selected cases are not a nonempty subset of Dataset508 fold-train")

        _, target_train, target_val = self._target_split_identity()
        expected_source_hashes = expected_source_train_hashes | expected_source_val_hashes
        target_hashes = {hash_identifier(identifier) for identifier in target_train | target_val}
        overlap = expected_source_hashes.intersection(target_hashes)
        if overlap:
            raise RuntimeError("Dataset501 and Dataset508 split identifiers overlap; verify cohort independence")

        actual_fingerprint = module_fingerprint(self.network.encoder)
        if actual_fingerprint != artifact["source_encoder_fingerprint"]:
            raise RuntimeError(
                "encoder tensors do not match the calibrated pretrained checkpoint; "
                "ensure official pretrained loading ran before CKPD setup"
            )
        if artifact["source_encoder_semantic_signature"] != encoder_semantic_signature(self.network):
            raise RuntimeError("calibration source encoder semantic signature differs from the target encoder")
        target_patch_size = getattr(getattr(self, "configuration_manager", None), "patch_size", None)
        if (not isinstance(target_patch_size, (tuple, list)) or len(target_patch_size) != 2
                or any(type(value) is not int or value <= 0 for value in target_patch_size)):
            raise RuntimeError("target resolved 2d patch_size is unavailable or invalid")
        if artifact["patch_size"] != list(target_patch_size):
            raise RuntimeError("calibration patch_size differs from the target resolved 2d configuration")
        convs = encoder_conv_layers(self.network)
        if artifact["encoder_conv_metadata"] != self._encoder_conv_metadata(convs):
            raise RuntimeError("calibration Conv2d names/shapes/dtypes differ from the target encoder")
        selected = artifact["selected_layers"]
        factors = artifact["factors"]
        if (not isinstance(selected, list) or len(selected) != int(artifact["layer_count"])
                or len(selected) != len(set(selected)) or set(selected) != set(factors)):
            raise RuntimeError("calibration selected-layer and factor manifests disagree")
        if int(artifact["rank"]) < 1 or int(artifact["rank"]) != self.CKPD_DEFAULT_RANK:
            raise RuntimeError("calibration rank differs from the approved rank-4 experiment")
        if int(artifact["layer_count"]) != self.CKPD_DEFAULT_LAYER_COUNT:
            raise RuntimeError("calibration layer count differs from the approved four-layer experiment")
        for name in selected:
            if name not in convs:
                raise RuntimeError(f"calibration selected unknown encoder Conv2d: {name}")
            self._validate_factor_payload(name, factors[name], convs[name], expected_weight=convs[name].weight)
        return artifact, artifact_sha, source_sha

    @staticmethod
    def _encoder_conv_metadata(convs: Mapping[str, torch.nn.Conv2d]) -> dict:
        return {
            name: {
                "shape": list(module.weight.shape),
                "dtype": str(module.weight.dtype),
                "stride": list(module.stride),
                "padding": list(module.padding),
                "dilation": list(module.dilation),
            }
            for name, module in sorted(convs.items())
        }

    @staticmethod
    def _validate_factor_payload(name: str, factor: object, conv: torch.nn.Conv2d,
                                 expected_weight: torch.Tensor | None = None) -> None:
        if not isinstance(factor, dict) or not {"base_weight", "A", "B", "singular_values", "asr", "damping"}.issubset(factor):
            raise RuntimeError(f"malformed CKPD factors for {name}")
        base = factor["base_weight"]
        A = factor["A"]
        B = factor["B"]
        singular = factor["singular_values"]
        if not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in (base, A, B, singular)):
            raise RuntimeError(f"non-finite CKPD factor for {name}")
        dimension = conv.weight.numel() // conv.out_channels
        rank = int(A.shape[0]) if A.ndim == 2 else -1
        if (tuple(base.shape) != tuple(conv.weight.shape) or A.ndim != 2
                or tuple(A.shape) != (rank, dimension) or rank != nnUNetTrainerCKPDTopK10EarlyStopping.CKPD_DEFAULT_RANK
                or tuple(B.shape) != (conv.out_channels, rank) or singular.ndim != 1):
            raise RuntimeError(f"invalid CKPD factor dimensions for {name}")
        if (expected_weight is not None and not torch.allclose(
                base.reshape(conv.out_channels, -1) + B @ A,
                expected_weight.detach().cpu().to(B.dtype).reshape(conv.out_channels, -1),
                atol=5e-4, rtol=5e-4)):
            raise RuntimeError(f"CKPD factors do not reconstruct the loaded pretrained kernel for {name}")
        if not isinstance(factor["asr"], (int, float)) or not torch.isfinite(torch.tensor(float(factor["asr"]))):
            raise RuntimeError(f"invalid ASR for {name}")
        if not isinstance(factor["damping"], (int, float)) or float(factor["damping"]) < 0:
            raise RuntimeError(f"invalid damping for {name}")

    @staticmethod
    def _to_decomposition(factor: Mapping) -> Conv2dDecomposition:
        return Conv2dDecomposition(
            base_weight=factor["base_weight"],
            A=factor["A"],
            B=factor["B"],
            singular_values=factor["singular_values"],
            asr=float(factor["asr"]),
            damping=float(factor["damping"]),
        )

    def _prepare_fresh_run(self) -> None:
        self._require_supported_network(self.network, bool(self.is_ddp), bool(self._do_i_compile()))
        self._assert_target_split_unchanged()
        artifact, artifact_sha, source_sha = self._load_and_validate_calibration()
        factors = {name: self._to_decomposition(value) for name, value in artifact["factors"].items()}
        apply_ckpd_adapters(self.network, factors)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self._ckpd_selected_layers = list(artifact["selected_layers"])
        self._ckpd_artifact_sha256 = artifact_sha
        self._ckpd_source_sha256 = source_sha
        self._ckpd_encoder_fingerprint = artifact["source_encoder_fingerprint"]
        self._ckpd_singular_values = {
            name: value["singular_values"].detach().cpu().clone()
            for name, value in artifact["factors"].items()
        }
        self._ckpd_asr = {name: float(value["asr"]) for name, value in artifact["factors"].items()}
        self._ckpd_damping = {name: float(value["damping"]) for name, value in artifact["factors"].items()}
        self._ckpd_adapters_installed = True

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # nnU-Net calls network.train() before each epoch; keep encoder buffers frozen.
        if self._ckpd_adapters_installed:
            self.network.encoder.eval()

    def run_training(self):
        if not self._ckpd_adapters_installed:
            if self._ckpd_resumed:
                raise RuntimeError("resumed CKPD topology was not restored")
            if getattr(self, "_ckpd_prior_run_artifacts", False):
                raise RuntimeError("refusing to start a fresh CKPD run in a folder with existing run artifacts")
            self._prepare_fresh_run()
        self._assert_target_split_unchanged()
        return super().run_training()

    def _training_state(self) -> dict:
        if not self._ckpd_adapters_installed:
            raise RuntimeError("cannot save a CKPD checkpoint before adapter installation")
        factors = {}
        modules = dict(self.network.named_modules(remove_duplicate=False))
        for name in self._ckpd_selected_layers:
            module = modules[name]
            if not parametrize.is_parametrized(module, "weight"):
                raise RuntimeError(f"CKPD adapter disappeared from {name}")
            adapter = module.parametrizations.weight[0]
            if not isinstance(adapter, CKPDWeightParametrization):
                raise RuntimeError(f"unexpected weight parametrization at {name}")
            factors[name] = {
                "base_weight": module.parametrizations.weight.original.detach().cpu().clone(),
                "A": adapter.A.detach().cpu().clone(),
                "B": adapter.B.detach().cpu().clone(),
                "singular_values": getattr(self, "_ckpd_singular_values", {}).get(name, torch.empty(0)).detach().cpu().clone(),
                "asr": float(getattr(self, "_ckpd_asr", {}).get(name, 0.0)),
                "damping": float(getattr(self, "_ckpd_damping", {}).get(name, 0.0)),
            }
        return {
            "schema_version": CKPD_SCHEMA_VERSION,
            "selected_layers": list(self._ckpd_selected_layers),
            "rank": self.CKPD_DEFAULT_RANK,
            "layer_count": self.CKPD_DEFAULT_LAYER_COUNT,
            "factors": factors,
            "source_checkpoint_sha256": self._ckpd_source_sha256,
            "calibration_sha256": self._ckpd_artifact_sha256,
            "encoder_fingerprint": self._ckpd_encoder_fingerprint,
            "target_split_sha256": self._ckpd_target_split_sha256,
            "target_plans_sha256": self._ckpd_target_plans_sha256,
            "scheduler_state": deepcopy(self.lr_scheduler.state_dict()),
            "rng_state": capture_rng_state(),
        }

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return
        if self.is_ddp or isinstance(self.network, OptimizedModule):
            raise RuntimeError("CKPD checkpoint export requires an unwrapped single-device network")
        checkpoint = {
            "network_weights": merged_state_dict(self.network),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "early_stopping_state": self._early_stopping_checkpoint_state(),
            self.CKPD_TRAINING_STATE_KEY: self._training_state(),
        }
        atomic_torch_save(checkpoint, filename)

    def _validate_public_weights(
        self, weights: object, network: torch.nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        if not isinstance(weights, Mapping):
            raise RuntimeError("CKPD checkpoint network_weights must be a tensor mapping")
        network = self.network if network is None else network
        expected = network.state_dict()
        if set(weights) != set(expected):
            missing = sorted(set(expected) - set(weights))[:5]
            extra = sorted(set(weights) - set(expected))[:5]
            raise RuntimeError(f"merged inference state keys differ (missing={missing}, extra={extra})")
        clean = {}
        for name, target in expected.items():
            value = weights[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
                raise RuntimeError(f"merged inference tensor has invalid type/shape: {name}")
            if value.dtype != target.dtype:
                raise RuntimeError(f"merged inference tensor dtype differs: {name}")
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"merged inference tensor is non-finite: {name}")
            clean[name] = value
        alias_groups: dict[int, list[str]] = {}
        for name, tensor in network.state_dict(keep_vars=True).items():
            alias_groups.setdefault(id(tensor), []).append(name)
        for aliases in alias_groups.values():
            if len(aliases) > 1:
                reference = clean[aliases[0]]
                if any(not torch.equal(reference.detach().cpu(), clean[name].detach().cpu()) for name in aliases[1:]):
                    raise RuntimeError(f"merged inference aliases disagree: {aliases}")
        return clean

    def _validate_resume_optimizer(self, state: object, expected_shapes: list[tuple[int, ...]]) -> None:
        if not isinstance(state, Mapping) or not isinstance(state.get("param_groups"), list):
            raise RuntimeError("CKPD optimizer checkpoint is malformed")
        incoming_groups = state["param_groups"]
        if len(incoming_groups) != 1:
            raise RuntimeError("CKPD optimizer parameter-group count differs")
        flat_indices = [index for group in incoming_groups for index in group.get("params", [])]
        if len(flat_indices) != len(expected_shapes) or len(set(flat_indices)) != len(flat_indices):
            raise RuntimeError("CKPD optimizer parameter layout differs")
        for saved_id, saved in state.get("state", {}).items():
            if saved_id not in flat_indices or not isinstance(saved, Mapping):
                raise RuntimeError("CKPD optimizer state refers to an unknown parameter")
            shape = expected_shapes[flat_indices.index(saved_id)]
            for key, value in saved.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and tuple(value.shape) != shape:
                    raise RuntimeError(f"CKPD optimizer tensor shape mismatch for {key}")

    def _validate_training_payload(
        self, payload: object, network: torch.nn.Module | None = None
    ) -> dict:
        if not isinstance(payload, dict):
            raise RuntimeError("CKPD training checkpoint is missing its private resume state")
        required = {
            "schema_version", "selected_layers", "rank", "layer_count", "factors",
            "source_checkpoint_sha256", "calibration_sha256", "encoder_fingerprint",
            "target_split_sha256", "target_plans_sha256", "scheduler_state", "rng_state",
        }
        if not required.issubset(payload) or payload["schema_version"] != CKPD_SCHEMA_VERSION:
            raise RuntimeError("CKPD training checkpoint schema is incomplete or unsupported")
        if int(payload["rank"]) != self.CKPD_DEFAULT_RANK or int(payload["layer_count"]) != self.CKPD_DEFAULT_LAYER_COUNT:
            raise RuntimeError("CKPD checkpoint rank/layer count differs from this Trainer")
        digest, _, _ = self._target_split_identity()
        if payload["target_split_sha256"] != digest:
            raise RuntimeError("Dataset501 splits_final.json differs from the CKPD checkpoint")
        if payload["target_plans_sha256"] != self._target_plans_identity():
            raise RuntimeError("Dataset501 plans differ from the CKPD checkpoint")
        selected = payload["selected_layers"]
        factors = payload["factors"]
        convs = encoder_conv_layers(self.network if network is None else network)
        if (not isinstance(selected, list) or len(selected) != self.CKPD_DEFAULT_LAYER_COUNT
                or len(set(selected)) != len(selected) or set(selected) != set(factors)):
            raise RuntimeError("CKPD checkpoint adapter manifest is inconsistent")
        if not isinstance(payload["encoder_fingerprint"], str) or len(payload["encoder_fingerprint"]) != 64:
            raise RuntimeError("CKPD checkpoint encoder fingerprint is malformed")
        for name in selected:
            if name not in convs:
                raise RuntimeError(f"CKPD checkpoint selected an unknown encoder layer: {name}")
            self._validate_factor_payload(name, factors[name], convs[name])
        return payload

    def _parametrized_state_for_resume(
        self, public: Mapping[str, torch.Tensor], factors: Mapping,
        network: torch.nn.Module | None = None,
    ) -> dict:
        network = self.network if network is None else network
        current = network.state_dict()
        modules = dict(network.named_modules(remove_duplicate=False))
        selected_modules = {name: modules[name] for name in factors}
        aliases = {name: _module_aliases(network, module) for name, module in selected_modules.items()}
        alias_to_factor = {alias: name for name, paths in aliases.items() for alias in paths}
        transformed = {}
        for key in current:
            if key in public:
                transformed[key] = public[key]
                continue
            matched = False
            for alias, name in alias_to_factor.items():
                prefix = f"{alias}.parametrizations.weight."
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if suffix == "original":
                        transformed[key] = factors[name]["base_weight"]
                    elif suffix == "0.A":
                        transformed[key] = factors[name]["A"]
                    elif suffix == "0.B":
                        transformed[key] = factors[name]["B"]
                    else:
                        raise RuntimeError(f"unknown CKPD parametrization key: {key}")
                    matched = True
                    break
            if not matched:
                raise RuntimeError(f"cannot map merged checkpoint tensor into adapter network: {key}")
        if set(transformed) != set(current):
            raise RuntimeError("failed to reconstruct complete parametrized network state")
        for name, target in current.items():
            value = transformed[name]
            if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                raise RuntimeError(f"CKPD resume tensor shape/dtype mismatch: {name}")
        return transformed

    def _validate_standard_resume_metadata(self, checkpoint: Mapping) -> None:
        required = {
            "trainer_name", "current_epoch", "logging", "_best_ema", "init_args",
            "inference_allowed_mirroring_axes", "early_stopping_state", "grad_scaler_state",
        }
        if not required.issubset(checkpoint):
            raise RuntimeError("CKPD checkpoint lacks standard nnU-Net resume metadata")
        if checkpoint["trainer_name"] != self.__class__.__name__:
            raise RuntimeError("CKPD checkpoint trainer_name differs")
        epoch = checkpoint["current_epoch"]
        if type(epoch) is not int or not 0 <= epoch <= self.num_epochs:
            raise RuntimeError("CKPD checkpoint current_epoch is invalid")

        init_args = checkpoint["init_args"]
        if (not isinstance(init_args, Mapping)
                or init_args.get("configuration") != self.configuration_name
                or init_args.get("fold") != self.fold):
            raise RuntimeError("CKPD checkpoint init_args do not match this configuration/fold")
        if not isinstance(checkpoint["logging"], Mapping):
            raise RuntimeError("CKPD checkpoint logging state is malformed")

        def finite_or_none(value: object) -> bool:
            return value is None or (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
            )

        if not finite_or_none(checkpoint["_best_ema"]):
            raise RuntimeError("CKPD checkpoint _best_ema is invalid")
        axes = checkpoint["inference_allowed_mirroring_axes"]
        if axes is not None and (
            not isinstance(axes, (tuple, list))
            or any(type(axis) is not int or axis < 0 for axis in axes)
            or len(set(axes)) != len(axes)
        ):
            raise RuntimeError("CKPD checkpoint mirroring axes are malformed")
        early = checkpoint["early_stopping_state"]
        if not isinstance(early, Mapping) or not {
            "best_monitored_ema_dice", "epochs_without_improvement", "triggered"
        }.issubset(early):
            raise RuntimeError("CKPD checkpoint early-stopping state is malformed")
        if (not finite_or_none(early["best_monitored_ema_dice"])
                or type(early["epochs_without_improvement"]) is not int
                or early["epochs_without_improvement"] < 0
                or type(early["triggered"]) is not bool):
            raise RuntimeError("CKPD checkpoint early-stopping values are invalid")

        scaler_state = checkpoint["grad_scaler_state"]
        if self.grad_scaler is None:
            if scaler_state is not None:
                raise RuntimeError("CKPD checkpoint has GradScaler state but this run does not")
        else:
            if not isinstance(scaler_state, Mapping):
                raise RuntimeError("CKPD checkpoint GradScaler state is malformed")
            try:
                trial_scaler = deepcopy(self.grad_scaler)
                trial_scaler.load_state_dict(scaler_state)
            except Exception as exc:
                raise RuntimeError(f"CKPD checkpoint GradScaler state failed prevalidation: {exc}") from exc

    @staticmethod
    def _plain_candidate_network(live_network: torch.nn.Module) -> torch.nn.Module:
        """Clone the live topology and remove only CKPD weight parametrizations."""
        candidate = deepcopy(live_network)
        for module in list(candidate.modules()):
            if not parametrize.is_parametrized(module, "weight"):
                continue
            entries = module.parametrizations.weight
            if len(entries) != 1 or not isinstance(entries[0], CKPDWeightParametrization):
                raise RuntimeError("CKPD resume cannot remove an unexpected weight parametrization")
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
        return candidate

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        # The wrapper makes even failures after nnU-Net's superclass loader atomic.
        old_attributes = self.__dict__.copy()
        old_rng_state = capture_rng_state()
        try:
            self._load_checkpoint_transaction(filename_or_checkpoint)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(old_attributes)
            _restore_rng_state_direct(old_rng_state)
            raise

    def _load_checkpoint_transaction(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if isinstance(filename_or_checkpoint, (str, os.PathLike)):
            checkpoint = torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        if not isinstance(checkpoint, dict):
            raise RuntimeError("CKPD checkpoint must be a dictionary")

        # Validate serialized generators before initialize() can consume RNG or
        # replace any live Trainer state.
        payload_before_init = checkpoint.get(self.CKPD_TRAINING_STATE_KEY)
        try:
            validate_rng_state(payload_before_init.get("rng_state") if isinstance(payload_before_init, Mapping) else None)
        except Exception as exc:
            raise RuntimeError(f"CKPD checkpoint RNG state failed prevalidation: {exc}") from exc
        if not self.was_initialized:
            self.initialize()

        self._validate_standard_resume_metadata(checkpoint)
        self._require_supported_network(self.network, bool(self.is_ddp), bool(self._do_i_compile()))
        candidate = self._plain_candidate_network(self.network)
        payload = self._validate_training_payload(checkpoint.get(self.CKPD_TRAINING_STATE_KEY), candidate)
        public = self._validate_public_weights(checkpoint.get("network_weights"), candidate)
        if not isinstance(payload["scheduler_state"], Mapping):
            raise RuntimeError("CKPD scheduler state is malformed")

        factors = payload["factors"]
        modules = dict(candidate.named_modules(remove_duplicate=False))
        for name in payload["selected_layers"]:
            conv = modules[name]
            for alias in _module_aliases(candidate, conv):
                merged = public[f"{alias}.weight"]
                factor = factors[name]
                reconstructed = (
                    factor["base_weight"].reshape(conv.out_channels, -1) + factor["B"] @ factor["A"]
                )
                if not torch.allclose(
                    reconstructed.to(merged.dtype), merged.reshape(conv.out_channels, -1),
                    atol=5e-4, rtol=5e-4,
                ):
                    raise RuntimeError(f"CKPD checkpoint merged tensor disagrees with factors at {alias}")

        candidate.load_state_dict(public, strict=True)
        decompositions = {
            name: self._to_decomposition(factors[name]) for name in payload["selected_layers"]
        }
        apply_ckpd_adapters(candidate, decompositions)

        # Build the optimizer against the candidate while preserving the live
        # network reference. This also validates the saved parameter ordering.
        live_network = self.network
        try:
            self.network = candidate
            candidate_optimizer, candidate_scheduler = self.configure_optimizers()
        finally:
            self.network = live_network
        expected_shapes = [tuple(parameter.shape) for parameter in candidate_optimizer.param_groups[0]["params"]]
        self._validate_resume_optimizer(checkpoint.get("optimizer_state"), expected_shapes)
        trial_scheduler = deepcopy(candidate_scheduler)
        try:
            trial_scheduler.load_state_dict(payload["scheduler_state"])
            validate_rng_state(payload["rng_state"])
        except Exception as exc:
            raise RuntimeError(f"CKPD scheduler/RNG state failed prevalidation: {exc}") from exc

        transformed = self._parametrized_state_for_resume(public, factors, candidate)
        prepared = dict(checkpoint)
        prepared["network_weights"] = transformed

        # Switch all mutable training objects to isolated candidate instances;
        # old references remain available to the outer rollback wrapper.
        candidate_logger = deepcopy(self.logger)
        candidate_scaler = deepcopy(self.grad_scaler) if self.grad_scaler is not None else None
        self.network = candidate
        self.optimizer = candidate_optimizer
        self.lr_scheduler = candidate_scheduler
        self.logger = candidate_logger
        self.grad_scaler = candidate_scaler
        super().load_checkpoint(prepared)
        self.lr_scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload["rng_state"])
        self._ckpd_selected_layers = list(payload["selected_layers"])
        self._ckpd_source_sha256 = payload["source_checkpoint_sha256"]
        self._ckpd_artifact_sha256 = payload["calibration_sha256"]
        self._ckpd_encoder_fingerprint = payload["encoder_fingerprint"]
        self._ckpd_target_split_sha256 = payload["target_split_sha256"]
        self._ckpd_target_plans_sha256 = payload["target_plans_sha256"]
        self._ckpd_singular_values = {
            name: factors[name]["singular_values"].detach().cpu().clone() for name in factors
        }
        self._ckpd_asr = {name: float(factors[name]["asr"]) for name in factors}
        self._ckpd_damping = {name: float(factors[name]["damping"]) for name in factors}
        self._ckpd_adapters_installed = True
        self._ckpd_resumed = True
