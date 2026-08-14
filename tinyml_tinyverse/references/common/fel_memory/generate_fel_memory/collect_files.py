import shutil
import subprocess
from pathlib import Path

home_dir = Path.home()
tools_dir = home_dir / "tools"
projects_dir = home_dir / "tiny-ml" / "individual-repos" / "tinyml-tinyverse" / "tinyml_tinyverse" / "references" / "common" / "fel_memory"

AM13_SDK_PATH = tools_dir / "am13e230x_sdk_26_00_00_06"
MSPM0_SDK_PATH = tools_dir / "mspm0_sdk_2_10_00_04"
C28_SDK_PATH = tools_dir / "C2000Ware_26_01_00_00"
C29_SDK_PATH = tools_dir / "f29h85_sdk_1"

AM13_DEST = projects_dir / "am13"
MSPM0_DEST = projects_dir / "mspm0"
C28_DEST = projects_dir / "c28"
C29_DEST = projects_dir / "c29"

def c28():
    C28_DRIVERLIB = C28_SDK_PATH / "driverlib"
    C28_DEVICE_SUPPORT = C28_SDK_PATH / "device_support"
    C28_LIB = C28_SDK_PATH / "libraries"

    for device_dir in C28_DRIVERLIB.iterdir():
        device_name = device_dir.name

        src_driverlib = device_dir / "driverlib"
        dst_device = C28_DEST / "driverlib" / device_name
        dst_fpu = C28_DEST / "fpu32"
        dst_driverlib = dst_device / "driverlib"
        src_support = C28_DEVICE_SUPPORT / device_name / "common"
        dsp_h = C28_LIB / "dsp" / "FPU" / "c28" / "include"
        cfft_eabi = C28_LIB / "dsp" / "FPU" / "c28" / "lib"
        fastrts_h = C28_LIB / "math" / "FPUfastRTS" / "c28" / "include"
        ai = C28_LIB / "ai" / "common" / "feature_extract"

        required_files = [
            src_driverlib,
            src_support / "include" / "driverlib.h",
            src_support / "include" / "device.h",
            dsp_h / "fpu32" / "fpu_fft_hann.h",
            dsp_h / "fpu32" / "fpu_cfft.h",
            dsp_h / "dsp.h",
            cfft_eabi / "c28x_fpu_dsp_library_eabi.lib",
            fastrts_h / "C28x_FPU_FastRTS.h",
            ai / "feature_extract_c28.c",
            ai / "feature_extract.c",
            ai / "feature_extract.h",
        ]

        if not all(f.exists() for f in required_files) or dst_driverlib.exists():
            print(list(f.exists() for f in required_files))
            continue

        shutil.copytree(src_driverlib, dst_driverlib)
        dst_device.mkdir(parents=True, exist_ok=True)
        dst_fpu.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(src_support / "include" / "driverlib.h", dst_driverlib / "driverlib.h")
        shutil.copy(src_support / "include" / "device.h", dst_device / "device.h")
        shutil.copy(ai / "feature_extract.c", C28_DEST / "feature_extract.c")
        shutil.copy(cfft_eabi / "c28x_fpu_dsp_library_eabi.lib", C28_DEST / "fel.a")
        shutil.copy(ai / "feature_extract.h", C28_DEST / "feature_extract.h")
        shutil.copy(ai / "feature_extract_c28.c", C28_DEST / "feature_extract_c28.c")

        subprocess.run(["python", "expand_headers.py", str(dsp_h / "dsp.h"), str(C28_DEST / "dsp.h")], capture_output=True)
        subprocess.run(["python", "expand_headers.py", str(dsp_h / "fpu32" / "fpu_fft_hann.h"), str(dst_fpu / "fpu_fft_hann.h")], capture_output=True)
        subprocess.run(["python", "expand_headers.py", str(dsp_h / "fpu32" / "fpu_cfft.h"), str(dst_fpu / "fpu_cfft.h")], capture_output=True)
        subprocess.run(["python", "expand_headers.py", str(fastrts_h / "C28x_FPU_FastRTS.h"), str(dst_fpu / "C28x_FPU_FastRTS.h")], capture_output=True)
        subprocess.run(["python", "expand_headers.py", str(dst_driverlib / "driverlib.h"), str(dst_device / "driverlib.h")], capture_output=True)

        for pattern in ["*.c", "*.h"]:
            for f in dst_driverlib.glob(pattern):
                f.unlink()

        for remove_path in [dst_driverlib]:
            if remove_path.exists():
                if remove_path.is_dir():
                    shutil.rmtree(remove_path)
                else:
                    remove_path.unlink()

def mspm0():
    MSPM0_SDK_SOURCE = MSPM0_SDK_PATH / "source"
    MSPM0_CMSIS = MSPM0_SDK_SOURCE / "third_party" / "CMSIS"
    MSPM0_FEATURE_EXTRACT = MSPM0_SDK_SOURCE / "ti" / "edgeAI" / "feature_extract"

    required_files = [
        MSPM0_CMSIS,
        MSPM0_FEATURE_EXTRACT,
    ]
    if not all(f.exists() for f in required_files):
        return

    shutil.copytree(MSPM0_CMSIS / "Core" / "Include", MSPM0_DEST / "CMSIS" , dirs_exist_ok=True)
    shutil.copytree(MSPM0_CMSIS / "DSP" / "Include", MSPM0_DEST / "CMSIS" , dirs_exist_ok=True)
    shutil.copy(MSPM0_CMSIS / "DSP" / "lib" / "ticlang" / "m0p" / "arm_cortexM0l_math.a", MSPM0_DEST)
    shutil.copytree(MSPM0_FEATURE_EXTRACT, MSPM0_DEST, dirs_exist_ok=True)

    subprocess.run(["python", "expand_headers.py", str(MSPM0_DEST / "CMSIS" / "arm_const_structs.h"), str(MSPM0_DEST / "arm_const_structs.h"), str(MSPM0_DEST / "CMSIS")], capture_output=True)
    subprocess.run(["python", "expand_headers.py", str(MSPM0_DEST / "CMSIS" / "arm_math.h"), str(MSPM0_DEST / "arm_math.h"), str(MSPM0_DEST / "CMSIS")], capture_output=True)

    with open(MSPM0_DEST / "feature_extract.h", "r") as fp:
        content = fp.read().replace('model/', '')

    with open(MSPM0_DEST / "feature_extract.h", "w") as fp:
        fp.write(content)

    for remove_path in [MSPM0_DEST / "CMSIS"]:
        if remove_path.exists():
            if remove_path.is_dir():
                shutil.rmtree(remove_path)
            else:
                remove_path.unlink()
    mspm0_path = MSPM0_DEST
    fel_path = mspm0_path / "fel"

    fel_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mv", str(mspm0_path / "arm_cortexM0l_math.a"), str(fel_path / "arm_cortexM0l_math.a")], capture_output=False)
    subprocess.run("tiarmar x arm_cortexM0l_math.a", cwd=str(fel_path), capture_output=True, shell=True)
    subprocess.run("tiarmar r fel.a *.o", cwd=str(fel_path), capture_output=True, shell=True)
    subprocess.run(["mv", str(fel_path / "fel.a"), str(mspm0_path / "fel.a")], capture_output=True)
    subprocess.run("rm -rf *.o arm_cortexM0l_math.a", cwd=str(fel_path), capture_output=True, shell=True)
    subprocess.run("rm -rf fel", cwd=str(mspm0_path), capture_output=True, shell=True)


def am13():
    AM13_SDK_SOURCE = AM13_SDK_PATH / "source"
    AM13_BUILD_LIBS = AM13_SDK_PATH / "build" / "am13e230x" / "lib" / "Release"
    AM13_CMSIS = AM13_SDK_SOURCE / "cmsis"
    AM13_FEATURE_EXTRACT = AM13_SDK_SOURCE / "ai" / "feature_extract"

    required_files = [
        AM13_CMSIS,
        AM13_FEATURE_EXTRACT / "feature_extract.c",
        AM13_FEATURE_EXTRACT / "feature_extract.h",
        AM13_FEATURE_EXTRACT / "fpu_fft_hann.h",
        AM13_FEATURE_EXTRACT / "feature_extract_cmsis_dsp.c",
        AM13_BUILD_LIBS / "libCMSISDSP_m33_ti_arm_clang.a",
    ]
    if not all(f.exists() for f in required_files):
        return

    shutil.copytree(AM13_CMSIS / "Core" / "Include", AM13_DEST / "CMSIS" , dirs_exist_ok=True)
    shutil.copytree(AM13_CMSIS / "DSP" / "Include", AM13_DEST / "CMSIS" , dirs_exist_ok=True)
    shutil.copy(AM13_FEATURE_EXTRACT / "feature_extract.c", AM13_DEST / "feature_extract.c")
    shutil.copy(AM13_FEATURE_EXTRACT / "feature_extract.h", AM13_DEST / "feature_extract.h")
    shutil.copy(AM13_FEATURE_EXTRACT / "fpu_fft_hann.h", AM13_DEST / "fpu_fft_hann.h")
    shutil.copy(AM13_FEATURE_EXTRACT / "feature_extract_cmsis_dsp.c", AM13_DEST / "feature_extract_am13.c")
    shutil.copy(AM13_BUILD_LIBS / "libCMSISDSP_m33_ti_arm_clang.a", AM13_DEST / "fel.a")

    subprocess.run(["python", "expand_headers.py", str(AM13_DEST / "CMSIS" / "arm_const_structs.h"), str(AM13_DEST / "arm_const_structs.h"), str(AM13_DEST / "CMSIS")], capture_output=True)
    subprocess.run(["python", "expand_headers.py", str(AM13_DEST / "CMSIS" / "arm_math.h"), str(AM13_DEST / "arm_math.h"), str(AM13_DEST / "CMSIS")], capture_output=True)

    with open(AM13_DEST / "feature_extract.h", "r") as fp:
        content = fp.read().replace('artifacts/', '')

    with open(AM13_DEST / "feature_extract.h", "w") as fp:
        fp.write(content)

    with open(AM13_DEST / "feature_extract.c", "r") as fp:
        content = fp.read().replace('artifacts/', '')

    with open(AM13_DEST / "feature_extract.c", "w") as fp:
        fp.write(content)

    for remove_path in [AM13_DEST / "CMSIS"]:
        if remove_path.exists():
            if remove_path.is_dir():
                shutil.rmtree(remove_path)
            else:
                remove_path.unlink()

# c28()
# mspm0()
# am13()