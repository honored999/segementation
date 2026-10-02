# ISLES2022 DWI → nnU-Net Dataset508

本脚本仅复制已下载并核对官方 MD5 的 ISLES2022 三维 DWI 与官方标签。
下载和官方 MD5 核对由调用者完成；脚本不下载、不访问服务器、不训练。
依赖：Python、SimpleITK、NumPy、Nibabel（建议使用服务器已有 newconda）。

## Windows PowerShell

将独立脚本 convert_isles2022_to_nnunet.py 复制到服务器，运行：

```powershell
conda run -n newconda python .\convert_isles2022_to_nnunet.py --source-root 'C:\lijialin\segementation\data\ISLES2022\source\ISLES-2022' --output-dir 'C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset508_ISLES2022DWI' --check
```

确认退出码为 0、status 为 checked 后，移除 --check 再运行相同命令。
默认 --expected-cases 250；该参数用于明确病例数，小型合成测试可改为 1。
--check 仍检查输出冲突和完整像素/几何，但不创建输出或父目录。

## 输入与检查

每例必须严格符合以下相对路径（病例号保留四位）：

- sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz
- derivatives/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_msk.nii.gz

DWI 与标签病例集合必须相同且数量正确；其他位置出现 DWI/标签或非预期命名会报错。
仅接受三维标量影像和有限像素值；先检查有限性，再对 Nibabel 应用 NIfTI
slope/inter 后的标签像素检查接近 0 或 1：绝对容差 atol=1e-6，rtol=0。
0.5、2、NaN、Inf 及超出容差的近 0/近 1 值仍拒绝。容差仅用于标签值验证，
不二值化、不重存标签。size、spacing、origin、direction 必须精确一致，几何不设容差。
Nibabel 检查 NIfTI 像素（避免 SimpleITK 将 NaN/Inf 读取为 0 导致漏检），
SimpleITK 检查几何；不自动修正。目录递归扫描拒绝越界链接和重复目录别名/循环。
其他模态不用于转换。全部病例验证且源 SHA256 稳定后才创建输出。

解析后的目标目录 basename 必须精确为 Dataset508_ISLES2022DWI（区分大小写）。
全部 DWI/标签的文件身份必须唯一，包括同病例、跨病例或角色的硬链接和路径别名；
使用文件系统身份而非内容哈希，身份独立但字节相同的文件允许通过。
拒绝已存在输出（含链接）、同父目录任何 Dataset508_* 条目（不区分大小写）、
解析后的源/输出祖先或后代重叠。请勿在扫描、验证或复制期间修改源文件、链接或目标父目录。
路径复查和哈希能检测通常的变动，但并非抵抗恶意并发文件系统修改的事务锁。

## 输出与失败处理

- imagesTr/ISLES_0001_0000.nii.gz
- labelsTr/ISLES_0001.nii.gz
- dataset.json：单通道 DWI；background=0、lesion=1；实际 numTraining；
  file_ending=.nii.gz；overwrite_image_reader_writer=SimpleITKIO。
- conversion_manifest.json：配对相对路径、解析后的根目录、每个源与输出 SHA256。

复制使用独占新建文件并逐字节读写，不切片、重采样、二值化或改写 NIfTI。
复制内容、输出重读 SHA256 和源重读 SHA256 必须一致。dataset.json 最后生成。
CLI 成功时退出码为 0；错误为非零，中断也不会报告成功。
只有退出码 0 且 status=complete 才代表完整转换；不能仅凭目录/清单存在判断。

复制、元数据写入失败或中断会保留未完成输出，不覆盖、不删除、不自动续传。
先检查失败原因与未完成目录；由操作者手动将其移到 nnUNet_raw 父目录之外，
再使用不存在且无 Dataset508_* 同级冲突的目标重新运行。未完成目录不得用于训练。

不生成 splits_final.json，不更改 Dataset501 数据、plans 或固定五折。
源影像及标签始终只读；清单包含源路径，应作为数据侧产物管理，不提交到代码仓库。

## 本地验证边界

测试仅使用微型合成 NIfTI，覆盖复制/哈希/源不变、数量与配对、几何、非法像素、
维度、输出冲突、链接边界、失败保留及 --check 无写入。
运行测试前检查 CPU、RAM、GPU、VRAM，任何达到 80% 时按资源策略降级。
本地合成测试不是服务器真实 250 例数据验证，也不是正式实验或临床结果。
独立 Level3 审查由主代理另行安排；服务器应先运行完整 --check 再转换。

缩放回归使用真实保存并重载的 int16 NIfTI：底层值 -32768/32767，
slope=1.5259021893143654e-05，inter=0.5000076293945312，
缩放值为 0/0.9999999997671694；覆盖 --check、转换字节与 SHA256 保持，
并通过已安装 nnU-Net SimpleITKIO.read_seg 检查背景、前景及体素位置。
该微型样例的读取结果不能替代服务器全部病例的预检；如读取路径丢失前景，
必须报告证据并停止协议变更，不自行改写原始 mask。
