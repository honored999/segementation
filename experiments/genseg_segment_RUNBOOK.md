# GenSeg epoch80 图像侧 S 推理与七指标 RUNBOOK（2026-10-23）

本续作补齐独立 `experiments.genseg_segment` 的 `audit`、`predict`、`accept` 三个入口。
当前是工程实现及 tiny CPU 证据；真实服务器 metadata、权重恢复、患者/HD-BET/CUDA/预测/评分均 NOT_RUN。
服务器实际 `use_mask_for_norm`、normalizer、resampler、geometry、可用资源均 NOT_VERIFIED，不能以本机参考 plans 代替。
先由主代理手动安排新增图像预处理/几何/输出链的独立聚焦只读复审，再另行授权同步代码。
本次不重训、不延长80epoch预算、不续训、不commit/push、不改原指标/训练源码/权重/记忆。
旧 `genseg_segment_HANDOFF.md` 保留为历史证据；本次另有 `genseg_segment_CONTINUATION_HANDOFF.md`。

## 冻结协议 v1

协议标识 `genseg_epoch80_image_only_v1_20261023`。outer19名单由原始固定fold0校验，禁止调参。
先审计80轮完成回执、实际plane数、总更新与末轮progress、联合schema、训练source与split、target manifest、baseline和实际plans。
checkpoint SHA直接取固定RUN的complete.json，交叉核验joint.pt；可选显式SHA也必须与两者一致。
无须另找独立秘密hash。元数据绑定对象只提供训练身份给原audit_joint，不构造TargetDataset、不读训练数组/GT；
历史targets由原固定manifest SHA绑定，不声称重建/重验全部76个训练target数组。
完整联合payload仍由原validate_payload校验G/D/S、Adam、alpha、BN，CPU weights_only同descriptor前后SHA；只构造S。

逐患者：绑定reader读取原始DWI与properties → 实际DefaultPreprocessor.run_case_npy(seg=None)，
按原transpose/crop/normalize/data及seg resampling顺序 → 对原DWI的LAS副本运行原HD-BET2.0.1/同权重SHA/默认TTA，
保持nnU-Net输入路径独立 → mask按原生切片平面扩张3mm（空片全保留，空卷拒绝），使用原targets映射函数 →
keep外填训练manifest的baseline min → checkpoint保存的inner61 scaling（不重拟合/不clipping）→
图像侧重采样0/-1 mask>=0清理 → 自有派生缓存（npz+JSON，无pickle/class_locations）。逐例释放数组。
HD-BET阶段结束后才将S放入GPU，逐例加载缓存、底/右补0至256倍数、eval/inference_mode、sigmoid>=0.5，所有切片都处理。
无S推理TTA、阈值搜索和连通域后处理。HD-BET默认TTA与此不同。

先二值化，再用绑定seg resampler逆重采样→逆crop→逆轴序；匹配reader的writer保存raw properties，
writer重读检验实际标签数组、binary、shape/spacing/origin/direction，不贴raw affine。
完成门禁重查恰好19例、无额外NIfTI、raw/pred/derived-input/mask哈希、各例JSON、完整geometry及S状态不变。
最后才写complete，写后再次验证；失败/中断移除成功标记并保存partial_cases，记录失败本身不替换原异常。
`accept`独立重新核验所有文件/源身份后才允许读取GT几何，不执行评分、不修改预测。

明确差异：训练监督三类seg>=0被seg=None图像侧0/-1替代，重采样边界及（实际开启时）masked-normalization统计区域
不保证逐像素等价。这不是GT泄漏，也不证明模型有效或无效。保留历史76例预处理统计来源；outer19是fold0开发评估，不是独立测试。
实际reader非SimpleITKIO、preprocessor非DefaultPreprocessor、normalizer无法支持等，具体失败关闭，不改plans或猜测后端。

## 输出与资源

PRED、CACHE、METRICS必须是互相隔离的新目录；已有目录或位于旧实验回执目录内部均拒绝。
不自动删除/续跑/覆盖；本版本一律拒绝旧CACHE，不启用未经审计的缓存复用。失败后保留目录，下次改trial号。
保护repo、raw/source、run/checkpoint、targets/baseline及其派生旧目录。
PRED含started.json、逐例NIfTI/JSON、resources.jsonl及最后complete.json或failed.json。
CACHE含每例HD-BET原DWI的LAS副本、原生mask、命令/日志/资源记录、prepared.npz、properties.json及来源哈希。
这些都是独立派生输出，保留供accept核验，评分前不要挪动或清理。

复用stage-aware ResourceGuard，固定物理GPU1/UUID；启动CPU/RAM/GPUutil/global VRAM<80%，至少12GiB空闲VRAM、8GiB空闲RAM。
RAM显式保留model/case预算；这里的8/4GiB是操作预留，不是测得峰值或保证，CPU加载完整联合checkpoint也会占RAM。
HD-BET运行期每秒检查，S每片检查并记录GPUutil；运行期GPUutil只记录，CPU/RAM/global VRAM继续限制。
启动不等待、不重试、不干预其他健康任务。HD-BET资源失败时终止本任务自己启动的CLI，保留失败输出。
不安装依赖；服务器需要既有psutil/HD-BET2.0.1/nnunetv2 2.8.1等绑定运行环境，本机newconda缺psutil时用PowerShell预检；本机也无HD_BET，metadata测试使用明确的包元数据替身，不是HD-BET运行证据。

## 服务器CMD：保存为 .cmd 后逐步运行

先确保本次代码已由主代理验收并获准同步。以下是未来命令，本次没有访问服务器。
三个阶段必须同一固定run、同一协议、同一预测目录；任一步失败立即停止。

```bat
@echo off
call conda activate nnunet5090
if errorlevel 1 exit /b 1
cd /d C:\lijialin\segementation\.worktrees\genseg-quality-repair
if errorlevel 1 exit /b 1
set "RUN=C:\lijialin\HD-BET_outputs\genseg_joint_adam_positive_epoch80_trial02"
set "SOURCE=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans_2d"
set "RAW=C:\lijialin\models3d\nnUNet\nnUNet_raw"
set "BASELINE=C:\lijialin\genseg30_derived_trial01\generator_5000steps_seed20261014_attempt02"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "OUT=C:\lijialin\HD-BET_outputs"
set "PRED=%OUT%\genseg_joint_adam_positive_epoch80_trial02_predictions19_trial01"
set "CACHE=%OUT%\genseg_joint_adam_positive_epoch80_trial02_maskcache19_trial01"
set "METRICS=%OUT%\genseg_joint_adam_positive_epoch80_trial02_metrics19_trial01"
set "IMAGES=%RAW%\Dataset501_StrokeLesion\imagesTr"
set "GT=%RAW%\Dataset501_StrokeLesion\labelsTr"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
if exist "%PRED%" exit /b 2
if exist "%CACHE%" exit /b 2
if exist "%METRICS%" exit /b 2

rem CPU-only metadata审计前预检：所有GPU只读查询；超限停止，不杀其他任务。
python -c "import psutil,subprocess,csv; cpu=psutil.cpu_percent(.5); ram=psutil.virtual_memory().percent; r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True); print('CPU',cpu,'RAM',ram,r.stdout,flush=True); rows=list(csv.reader(r.stdout.splitlines())); assert cpu<80 and ram<80 and rows and all(float(v[1])<80 and 100*float(v[2])/float(v[3])<80 for v in rows), 'resource guard refused'"
if errorlevel 1 exit /b 3
python -m experiments.genseg_segment audit --run "%RUN%" --source "%SOURCE%" --raw-root "%RAW%" --targets "%TARGETS%" --baseline "%BASELINE%"
if errorlevel 1 exit /b 4

rem 预测内部重新核对metadata和资源，固定物理GPU1；实际绑定信息写started/complete。
python -m experiments.genseg_segment predict --run "%RUN%" --source "%SOURCE%" --raw-root "%RAW%" --targets "%TARGETS%" --baseline "%BASELINE%" --output "%PRED%" --mask-cache "%CACHE%" --model-reserve-gib 8 --case-reserve-gib 4
if errorlevel 1 exit /b 5

rem 独立评分前验收：不只检查complete存在。无预测错误后才只读GT几何。
python -c "import psutil,subprocess,csv; cpu=psutil.cpu_percent(.5); ram=psutil.virtual_memory().percent; r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True); print('CPU',cpu,'RAM',ram,r.stdout,flush=True); rows=list(csv.reader(r.stdout.splitlines())); assert cpu<80 and ram<80 and rows and all(float(v[1])<80 and 100*float(v[2])/float(v[3])<80 for v in rows), 'resource guard refused'"
if errorlevel 1 exit /b 6
python -m experiments.genseg_segment accept --output "%PRED%" --raw-images "%IMAGES%" --gt-dir "%GT%" --metrics-output "%METRICS%"
if errorlevel 1 exit /b 7

rem 使用原指标脚本，指标目录必须仍不存在，预测不可再编辑。
if exist "%METRICS%" exit /b 8
python -c "import psutil,subprocess,csv; cpu=psutil.cpu_percent(.5); ram=psutil.virtual_memory().percent; r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True); print('SCORE CPU',cpu,'RAM',ram,r.stdout,flush=True); rows=list(csv.reader(r.stdout.splitlines())); assert cpu<80 and ram<80 and rows and all(float(v[1])<80 and 100*float(v[2])/float(v[3])<80 for v in rows), 'resource guard refused'"
if errorlevel 1 exit /b 9
python evaluate_segmentation_metrics.py --pred-dir "%PRED%" --gt-dir "%GT%" --output-dir "%METRICS%" --f2-mode paper
if errorlevel 1 exit /b 10
```

`accept`与原评分之间应紧接执行，不编辑/替换预测或GT。原脚本不内建19例门禁，不能绕过accept。
输出原三份文件：case_metrics.csv、summary_metrics.csv、summary_metrics.json。
病例macro七指标 Dice/IoU/F2/AVD/LCD/Recall/HD95；逐指标NaN与valid_cases沿原定义披露，不能把缺失当0。
F2 paper定义5TP/(5TP+4FP+FN)，LCD为26连通病灶数量绝对差，非病灶检出率。

## 若metadata缺失或不支持：精确只读导出

本地无法确认服务器文件是否存在；当前无已证实的缺字段。入口对缺文件/identity/backend不匹配给具体异常，禁止补造或要求重训。
优先提供实际异常以及下面元数据，不需要患者数组/标签内容：

```bat
type "%RUN%\complete.json"
type "%SOURCE%\..\nnUNetPlans.json"
type "%SOURCE%\..\dataset.json"
certutil -hashfile "%SOURCE%\..\splits_final.json" SHA256
certutil -hashfile "%RUN%\joint.pt" SHA256
certutil -hashfile "%TARGETS%\targets.json" SHA256
type "%TARGETS%\targets.json"
type "%BASELINE%\generator.json"
python -c "import importlib.metadata as m; from HD_BET.paths import folder_with_parameter_files; print({n:m.version(n) for n in ['torch','nnunetv2','hd-bet','numpy','scipy','SimpleITK']}); print('HD_BET_WEIGHTS_ROOT',folder_with_parameter_files)"
```

元数据可能包含本地病例标识/路径，仅在获准范围回传，不放入项目记忆或公开发布。
必须核对：actual configuration normalization_schemes/use_mask_for_norm/resamplers/reader/preprocessor/spacing/transpose，
训练receipt target_identity/source/epoch_progress/scaling，manifest baseline fill/HD-BET recipe/weights SHA。
实际不支持的后端保持拒绝，不能用本机false替代，也无需修改训练。

## 本次验证

仅newconda、串行tiny CPU；CUDA_VISIBLE_DEVICES=-1、OMP/MKL/OPENBLAS=1、PYTHONDONTWRITEBYTECODE=1。
每次调用前PowerShell短窗采样CPU、CIM RAM、nvidia-smi全部可见GPU；有>=80%则停止或降级，无安装依赖。
当前亲跑测试命令与最终结果、SHA和Git状态以续作HANDOFF为准；旧HANDOFF的9项不是本次证据。
原作者完整S结构节点未因本次未改恢复结构而重复；完整schema恢复/Adam/BN不变使用tiny fixture复核。
模拟GPU传输/HD-BET输出的19例测试不证明真实CUDA/HD-BET或生产checkpoint可运行。
Python3.10使用ast语法检查，不声称运行过3.10解释器。没有全仓/202项/高阶梯度测试、生产指标或质量结论。


## 2026-10-23 B1/B2 最小修复终结 HANDOFF

status: FIXED_FOCUSED_TESTS_PASS / PENDING_PARENT_ACCEPTANCE_AND_MANUAL_REVIEW。
角色：本对话 focused fixer；工具未提供可核实 Luna/DSH profile，不冒称该 profile。
任务书明确禁止创建代理并指定 fixer，因此当前对话直接修改源码/测试；未创建代理或审查。
工作树：`E:\study\研一\work14-图像分割\segementation\.worktrees\genseg-quality-repair\segementation`。
分支 `codex/genseg-quality-repair`；起止 HEAD 均为 `0e92c26125e5176f0838e6e66bc742d75487c630`。
起点三个 SHA256 与任务书一致，仅修改推理入口、原测试、本文档。两个历史 HANDOFF 保留。

B1：started/complete 新增 `cache_root`，值为 fresh_output 解析后的绝对规范 CACHE 根。
verify_predictions 必须有此字段，核对 CACHE/PRED 互不重叠；病例目录 resolve 后不得逃逸，
prepared/properties/mask 必须分别是 CACHE/case/prepared.npz、properties.json、hd_bet_bet.nii.gz。
原文件哈希检查保留。accept 先核验缓存归属，再将 CACHE 纳入原 protected 列表。
拒绝 CACHE、CACHE/metrics_new、包含 CACHE 的祖先；独立同级 metrics 允许，验收不写 metrics/GT。
测试使用生产 predict 回执，篡改根及三个内容相同但位于根外的文件路径，保持病例清单一致仍拒绝。
无 cache_root 的回执不提供兼容旁路。

B2：仅清理当前 Popen 子进程，terminate、15秒 wait、必要 kill、15秒回收各自受保护。
清理失败仍继续后续清理尝试；简短 `cleanup_errors` 列表只含 stage/error_type，诊断属性失败亦受保护。
最后裸 raise；失败回执保留原 error_type/reason/resource_snapshot 并附 cleanup_errors。
原异常对象、args、resource_snapshot 和原 traceback 节点均由回归核对；不继续病例，不生成 complete。
用生产 predict 失败路径注入 kill OSError，failed.json 仍为原 resource fault。

### 本轮亲跑命令与证据

环境：`D:\Anaconda\envs\newconda\python.exe`，串行 tiny CPU。
每次调用前 CIM CPU/RAM、nvidia-smi 查询；CUDA_VISIBLE_DEVICES=-1，
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1，PYTHONDONTWRITEBYTECODE=1。
首次 CIM 为即时采样，非短窗均值；未测峰值，不声称预测资源峰值。

公共命令前缀：
`D:\Anaconda\envs\newconda\python.exe -m pytest -q -p no:cacheprovider`
所有节点前缀：`experiments/genseg_joint/tests/test_segment.py::`。

1. RED：节点 test_actual_prediction_completion_and_scoring_gate 与 test_native_hd_command_receipt_and_resource_abort；
   `--basetemp=$env:TEMP/genseg_b1b2_red_20261023`。
   CPU 21%，RAM 65.7892%，GPU0 24%，VRAM 1010/6144 MiB。
   **6 failed, 2 passed, 4 warnings in 6.63s**；缺 cache_root；terminate/wait/kill/reap/生产失败路径覆盖原异常。
2. 首次修复：上述两节点加 test_prediction_failure_removes_complete_preserves_original；
   `--basetemp=$env:TEMP/genseg_b1b2_green_20261023`。
   CPU 28%，RAM 64.0589%，GPU0 43%，VRAM 1021/6144 MiB。
   **1 failed, 10 passed, 4 warnings in 10.08s**。
   B2 七参数和原失败回执三参数通过；B1 伪造病例根被 FileNotFoundError 拒绝，调整为先 is_dir 门禁 ValueError。
3. 仅 B1 重跑：`--basetemp=$env:TEMP/genseg_b1b2_b1_final_20261023`。
   CPU 32%，RAM 63.7430%，GPU0 7%，VRAM 1007/6144 MiB。
   **1 failed, 4 warnings in 6.05s**；新增测试误用只允许新建的 write_json 覆盖篡改清单。
   仅将测试篡改模拟改用 write_text，生产 write_json 不变。
4. 最终 B1：`--basetemp=$env:TEMP/genseg_b1b2_b1_pass_20261023`。
   CPU 34%，RAM 63.4160%，GPU0 25%，VRAM 992/6144 MiB。
   **1 passed, 4 warnings in 7.81s**。最终相关证据合计 11 个参数用例通过，非单次11项全绿运行。
   B1 最后一处生产变更仅将病例目录 is_dir 判断提前，故不重复已通过 B2/失败回执节点。
   四个 warning 均为 batchgenerators 导入 scipy.ndimage 旧命名空间的 DeprecationWarning。

最终源码 SHA256：`22d4e6515950608171ab2bd982b69c62ff9cae60c58f00f8c5a13696ab65cf64`。
最终测试 SHA256：`f10fae2aaa8ffb0c81cf3d8cc8f052843519609b05a9c4f92b35f8e6b11e3922`。
本文档自身 SHA256 以终端最终 Get-FileHash 为准（避免自引用）。

Git：tracked/index 均无 diff；仍为原五个 untracked；未暂存、commit、push。
空白检查使用 git diff --check 和 git diff --no-index --check -- /dev/null <三个文件>，结果见最终终端。
保护边界：主工作区、训练源码、metrics/raw/GT/split/targets/权重/旧输出、项目与全局记忆均未改。
测试产物仅为 TEMP 下 tiny synthetic；未运行服务器预测、真实 HD-BET/CUDA、患者数据、生产 metadata/真实 checkpoint、19例评分、全仓/202项/作者完整S/高阶梯度。
上述生产验证全部 NOT_RUN。本轮不作分割质量或正式医学实验结论。
LAS hash/HD-BET完成回执增强暂缓。主代理验收后由用户手动派发独立 B1/B2 复审，尚无独立 PASS。
