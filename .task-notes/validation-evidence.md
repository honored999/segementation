# Current implementation validation evidence

Target branch codex/upernet-no-stage7; baseline 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89. Implementation delegated to leaf worker 01a0ff74-8a5d-7082-b65d-e8d48de28517; initial usage-limit failure recovered by resuming the same worker and saved tests. No main-agent production/test edits.

Environment: D:/Anaconda/envs/newconda/python.exe; nnunetv2==2.8.1 confirmed by fresh official external resolver. CPU synthetic only, 2 threads, single-process pytest, no cache/bytecode, D:/codex-pytest-temp/upernet-no-stage7 temp root. Every shell call requested elevated approval. CPU/RAM/GPU/VRAM preflight before every test; all below 80 percent.

Final focused: python -m pytest nnunet_ext_trainers/tests/test_upernet_no_stage7.py -q -p no:cacheprovider --basetemp D:/codex-pytest-temp/upernet-no-stage7 -> 10 passed, 5 warnings; preflight CPU 18/29 percent, RAM 66 percent, GPU 25 percent, VRAM 733/6144 MiB.
Affected baseline: python -m pytest nnunet_ext_trainers/tests/test_official_upernet_trainers.py -q -p no:cacheprovider --basetemp D:/codex-pytest-temp/upernet-no-stage7 -> 25 passed, 7 warnings; preflight CPU 17/12 percent, RAM 69.3 percent, GPU 5 percent, VRAM 1063/6144 MiB.

Checks: 7-stage returned model, no stage7 parameters/hooks; selected (1,3,5,6); exact retained encoder weights/config; PPM (1,2,4), FPN128; unchanged training-policy inheritance; finite 128x128 tiny-channel CPU forward; exact convolution feature-map count vs actual hooks; strict state_dict reconstruction; actual PlansManager/ConfigurationManager immutability; unrelated-cwd fresh official external discovery/current builder signature. 512x512 native geometry calculated as 256,64,16,8; no full-size real-model/data forward claimed.

Current reviewed SHA256:
Trainer A691935AF445194D37A0BCA901F15B44D9CB07C9D3ECBDA2077306F86FD73DAF
Tests A45DA5D6F8B6B8950F3115038FCB6EE8443B542226615BAB079A4F32956B1405
README FD89AF0FB3C6006A0F55FBBA0DE4EE1401B147701129258D2C9FFB325A0E16E5

Independent Level3 read-only review PASS from leaf context 01a10098-6331-7173-b3c1-79d4c667b442; final three hashes match. No blockers; no repeated runtime tests. Branch-local memory updated; canonical master memory not modified because this branch is not integrated. No real-data/checkpoint/CUDA validation, training, clinical/formal performance conclusion or master integration. Source data and fixed splits untouched. Parent final inspection confirmed digests and PASS. User subsequently authorized branch commit/push for a same-name server worktree; tests and review remain bound to the unchanged implementation snapshot.
