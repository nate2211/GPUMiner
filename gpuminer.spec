# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_dir = Path(SPECPATH)

datas = [
    (str(project_dir / "blocknet_client.py"), "."),
    (str(project_dir / "blocknet_mining_backend.py"), "."),
    (str(project_dir / "blocknet_randomx_vm_opencl.cl"), "."),
    (str(project_dir / "cpu_verify.py"), "."),
    (str(project_dir / "gui.py"), "."),
    (str(project_dir / "main.py"), "."),
    (str(project_dir / "models.py"), "."),
    (str(project_dir / "opencl_miner.py"), "."),
    (str(project_dir / "solo_zmq.py"), "."),
    (str(project_dir / "stratum_client.py"), "."),
    (str(project_dir / "stratum_connection.py"), "."),
    (str(project_dir / "utils.py"), "."),
    (str(project_dir / "worker.py"), "."),
]

binaries = [
    (str(project_dir / "MiningProject.dll"), "."),
    (str(project_dir / "OpenCL.dll"), "."),
    (str(project_dir / "randomx-dll.dll"), "."),
]

hiddenimports = [
    "blocknet_client",
    "blocknet_mining_backend",
    "cpu_verify",
    "gui",
    "models",
    "opencl_miner",
    "solo_zmq",
    "stratum_client",
    "stratum_connection",
    "utils",
    "worker",
]

hiddenimports += collect_submodules("PyQt5")
hiddenimports += collect_submodules("pyopencl")

a = Analysis(
    ["main.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="GPUMiner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    onefile=True,
)