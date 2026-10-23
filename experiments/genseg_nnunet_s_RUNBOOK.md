# GenSeg nnU-Net 2D S trial — implementation / future server template

2026-10-23. 本轮只做本地合成CPU工程验证。服务器命令均未执行。
源码必须先由用户手动独立Level3审查，再由主代理决定同步；本轮无stage/commit/push。

## 固定实现边界

新策略 `joint_adam_positive_nnunet2d_sgdproxy_v1`、S kind `nnunet2d_plainconv_1logit_v1`、checkpoint schema3。
旧作者UNet入口及schema2仍默认、严格拒绝schema3；新reader也拒绝schema2。
训练与推理复用现有joint_runtime、RepairJointEngine、preview和genseg_segment，不复制训练或几何管线。

实际SOURCE的完整plans（通过PlansManager解析2d，包括继承配置）、architecture kwargs与导入项、dataset、plans/dataset/split SHA、安装版本与关键源码SHA绑定在S描述中。
check在meta device生成layout/参数/共享别名信息，不分配完整网络权重，也不forward。
不支持的实际plans明确报错，不回退本地reference plans、不升级依赖。
当前支持2D PlainConvUNet + plans InstanceNorm2d(track_running_stats=False)、无dropout、DWI单通道二分类。
保留官方初始化、层数、宽度、激活与norm参数；单logit sigmoid/BCE+foreground Dice、关闭深监督是明确适配。
这不是nnU-Net默认Trainer训练，不是两类softmax/CE，也不是GenSeg论文精确复现。

G/D沿用GD20权重/BN/两组alpha；S随机初始化且不载入旧Dataset501 S权重。
所有实际Adam和alpha moments、RNG、sampler全新；virtual G/S保持单步私有eval SGD；G/D实际BN计数仍1/3。
S只有IN、无BN与running buffers，S BN更新日志为0；实际S forward仍2次。
官方decoder即使deep_supervision=False仍注册闲置seg heads：它们保留在完整网络和去重optimizer参数组内，
只有实际forward不使用的heads明确允许grad=None且必须无Adam moments；其他参数断链或moment缺失均拒绝。
参数所有权按真实named_parameters去重，state_dict所有别名必须一致。
functional_call显式展开别名，并在finally恢复原Parameter对象，防共享模块引用污染live权重/optimizer所有权。

原95例split、inner61/15、outer19末轮开发评估、real阴性、positive-only fake、预处理/缩放/G/D/loss/rates不改。
输入只在右下padding到lcm(256,实际各轴stride乘积)，image=0、label=-1，不resize、不拿plans batch_size替换joint unit。
静态拒绝最深IN空间只有一个元素。既有resource/throughput/失败回执和support-v2保留。
只存末轮，无best、早停、resume。相同seed不保证跨架构随机/浮点轨迹一致。
环状纹理/黑孔问题未证明解决，完整网络真实高阶CUDA兼容/显存峰值UNKNOWN。

官方执行依据是实际安装源码，不跟随master升级：
- https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py
- https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

## 服务器CMD模板：逐阶段人工执行

以下是未来路径模板，不代表已创建/核实。保留既有reference/quality-repair checkout和所有旧输出。
主代理完成审查和授权同步后，确认新checkout含本次逐文件SHA对应源码。勿把未提交本地分支当作服务器已可pull。
在 `.cmd` 文件中运行各独立阶段；不把下面所有阶段合并自动连续运行。

公共变量（每个阶段保留；预算必须依据本机空闲资源人工输入，不是已验证上限）：

```bat
@echo off
call conda activate nnunet5090
if errorlevel 1 exit /b 1
set "CODE=C:\lijialin\segementation\.worktrees\genseg-nnunet2d-s-trial"
cd /d "%CODE%"
if errorlevel 1 exit /b 1
git branch --show-current
git rev-parse HEAD
git status --short
rem 人工核对分支/提交/源码SHA后再继续。不得reset/stash/覆盖现有修改。
set "PYTHONPATH=%CODE%;%PYTHONPATH%"
set "nnUNet_raw=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "nnUNet_preprocessed=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed"
set "SOURCE=%nnUNet_preprocessed%\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "GD20=C:\lijialin\HD-BET_outputs\genseg_repair_gd_epoch20_trial01"
set "GD20_SHA=1547b42e65a7f21e8452d3509ce862f059884dd7e2bb82047eea7b33b7a5b75b"
set "STRATEGY=joint_adam_positive_nnunet2d_sgdproxy_v1"
set "OUT=C:\lijialin\HD-BET_outputs"
set "PREFLIGHT=%OUT%\genseg_nnunet2d_s_preflight3_trial01"
set "JOINT=%OUT%\genseg_nnunet2d_s_epoch80_trial01"
set "PREVIEW=%OUT%\genseg_nnunet2d_s_epoch80_preview_trial01"
set "PRED=%OUT%\genseg_nnunet2d_s_epoch80_predictions19_trial01"
set "CACHE=%OUT%\genseg_nnunet2d_s_epoch80_maskcache19_trial01"
set "METRICS=%OUT%\genseg_nnunet2d_s_epoch80_metrics19_trial01"
set "IMAGES=%nnUNet_raw%\Dataset501_StrokeLesion\imagesTr"
set "GT=%nnUNet_raw%\Dataset501_StrokeLesion\labelsTr"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set /p "CACHE_BYTES=Explicit owned-cache budget bytes: "
set /p "LOADER_WORKSPACE_BYTES=Explicit external loader transient reserve bytes: "
set /p "MODEL_RESERVE_BYTES=Conservative model/Adam/private higher-order RAM reserve bytes: "
```

每次check/preflight/训练/审计/预测/preview/验收/评分前重新资源检查；缺telemetry或达到80%立即停止，不能杀其他任务腾资源。
训练/预测内部仍有原资源guard，运行期GPUutil记录而非误把自身忙碌判成外部抢占；CPU/RAM/global VRAM限制不变。
以下外部启动检查只查询计划使用的物理GPU1：

```bat
python -c "import psutil,subprocess,csv; cpu=psutil.cpu_percent(.5); ram=psutil.virtual_memory().percent; r=subprocess.run(['nvidia-smi','-i','1','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True); print('RESOURCE',cpu,ram,r.stdout,flush=True); rows=list(csv.reader(r.stdout.splitlines())); assert cpu<80 and ram<80 and rows and all(float(v[1])<80 and 100*float(v[2])/float(v[3])<80 for v in rows), 'resource guard refused'"
if errorlevel 1 exit /b 2
```

阶段1，实际SOURCE metadata/header-only check；fresh output路径仍不创建：

```bat
python -m experiments.genseg_joint.joint_cli check --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PREFLIGHT%" --gpu 1 --pretrained-run "%GD20%" --pretrained-sha256 "%GD20_SHA%" --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_WORKSPACE_BYTES% --model-reserve-bytes %MODEL_RESERVE_BYTES%
if errorlevel 1 exit /b 3
```

核对check的完整S descriptor、实际版本/SHA/strides/IN/padded_shapes以及cache+loader+model headroom；不以参数量推断5090能容纳。
阶段2，仅新目录3-unit单卡preflight，不恢复旧S；不得自动续跑：

```bat
if exist "%PREFLIGHT%" exit /b 4
python -m experiments.genseg_joint.joint_cli train --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PREFLIGHT%" --gpu 1 --pretrained-run "%GD20%" --pretrained-sha256 "%GD20_SHA%" --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_WORKSPACE_BYTES% --model-reserve-bytes %MODEL_RESERVE_BYTES% --steps 3
if errorlevel 1 exit /b 5
```

人工检查complete、三步actual Adam/alpha、S BN=0且无running buffers、G1/D3、finite/nonzero梯度、failure/timing/allocator与全卡峰值。
若double-backward/OOM/metadata失败，保留failed目录和具体错误，停止；不能改norm/虚拟优化器/缩层宽度绕过。
核实固定输入S保存读回eval一致，并检查G定性图像，仍不声称合成质量通过。

阶段3，只有用户看过真实preflight并决定运行后，另一个fresh目录从原GD20重新开始80完整real epochs；不是从preflight resume：

```bat
if exist "%JOINT%" exit /b 6
python -m experiments.genseg_joint.joint_cli train --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%JOINT%" --gpu 1 --pretrained-run "%GD20%" --pretrained-sha256 "%GD20_SHA%" --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_WORKSPACE_BYTES% --model-reserve-bytes %MODEL_RESERVE_BYTES% --epochs 80
if errorlevel 1 exit /b 7
```

只认final80 complete的checkpoint SHA，不用outer19选择权重/学习率/阈值。
从新JOINT的complete.json读取checkpoint_sha256，人工填入；不沿用旧80epoch SHA：

```bat
python -c "import json,os; print(json.load(open(os.path.join(os.environ['JOINT'],'complete.json'),encoding='utf-8'))['checkpoint_sha256'])"
if errorlevel 1 exit /b 8
set /p "JOINT_SHA=New completed final80 checkpoint SHA256: "
python -m experiments.genseg_segment audit --s-kind nnunet2d --run "%JOINT%" --checkpoint-sha256 "%JOINT_SHA%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --targets "%TARGETS%" --baseline "%BASELINE%"
if errorlevel 1 exit /b 9
```

可选独立G preview，不读outer19数组、不构建分配S网络；schema3的S layout仅meta检查：

```bat
if exist "%PREVIEW%" exit /b 10
python -m experiments.genseg_joint.joint_cli preview --strategy "%STRATEGY%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --baseline-run "%BASELINE%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%PREVIEW%" --gpu 1 --run "%JOINT%" --checkpoint-sha256 "%JOINT_SHA%" --patient-limit 4
if errorlevel 1 exit /b 11
```

阶段4，完整outer19 S-only预测，image-only support-v2、阈值sigmoid>=.5、原始空间、无GT输入、无TTA/搜索：

```bat
if exist "%PRED%" exit /b 12
if exist "%CACHE%" exit /b 12
if exist "%METRICS%" exit /b 12
set /p "MODEL_RESERVE_GIB=Conservative prediction model RAM reserve GiB: "
set /p "CASE_RESERVE_GIB=Conservative prediction preprocessing/HD-BET case RAM reserve GiB: "
python -m experiments.genseg_segment predict --s-kind nnunet2d --run "%JOINT%" --checkpoint-sha256 "%JOINT_SHA%" --source "%SOURCE%" --raw-root "%nnUNet_raw%" --targets "%TARGETS%" --baseline "%BASELINE%" --output "%PRED%" --mask-cache "%CACHE%" --model-reserve-gib %MODEL_RESERVE_GIB% --case-reserve-gib %CASE_RESERVE_GIB%
if errorlevel 1 exit /b 13
```

阶段5，重新resource检查后先exact19/hash/geometry/support验收，之后才读GT和原指标脚本；验收与评分紧接，不能中间编辑预测：

```bat
python -m experiments.genseg_segment accept --output "%PRED%" --raw-images "%IMAGES%" --gt-dir "%GT%" --metrics-output "%METRICS%"
if errorlevel 1 exit /b 14
if exist "%METRICS%" exit /b 15
python evaluate_segmentation_metrics.py --pred-dir "%PRED%" --gt-dir "%GT%" --output-dir "%METRICS%" --f2-mode paper
if errorlevel 1 exit /b 16
```

七指标及NaN/valid_cases口径不变，保留case_metrics.csv/summary_metrics.csv/summary_metrics.json。
outer19是fold0开发证据。架构替换包括深度/宽度/norm/激活/初始化差异，不可声称只比较卷积或解决G伪影。
