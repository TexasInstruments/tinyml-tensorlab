import os
import logging
from pathlib import Path
from argparse import ArgumentParser

logger = logging.getLogger("ti_fel_compiled")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s          :  %(message)s"
)

def setup_paths():
    """Setup common directory paths."""
    fel_memory_dir = Path(__file__).parent
    return fel_memory_dir


def cleanup_artifacts(cleanup_items):
    """Clean up build artifacts. Items can be glob patterns (*.o) or specific files."""
    for item in cleanup_items:
        if '*' in item:
            for f in Path(".").glob(item):
                f.unlink()
        else:
            p = Path(item)
            if p.exists():
                p.unlink()


def format_bytes(bytes_val):
    """Format bytes to KB."""
    return f"{bytes_val} bytes ({bytes_val / 1024:>7.2f} KB)"


def print_memory_report(memory_map, archive_totals, files_to_report):
    """Log formatted memory report."""
    total_code = 0
    total_ro_data = 0
    total_rw_data = 0

    logger.info("============================================================")
    logger.info("TI Feature Extraction Library Memory Usage")
    logger.info("============================================================")

    for file_key, display_name in files_to_report:
        if file_key in archive_totals:
            sizes = archive_totals[file_key]
        elif file_key in memory_map:
            sizes = memory_map[file_key]
        else:
            continue
        total_code += sizes[0]
        total_ro_data += sizes[1]
        total_rw_data += sizes[2]

    logger.info(f"Code:                      {format_bytes(total_code)}")
    logger.info(f"RO Data:                   {format_bytes(total_ro_data)}")
    logger.info(f"RW Data:                   {format_bytes(total_rw_data)}")
    total_bytes = total_code + total_ro_data + total_rw_data
    logger.info(f"Total:                     {format_bytes(total_bytes)}")
    logger.info("==============================================================")


def get_memory(map_file):
    memory_map = {}
    archive_totals = {}
    in_archive = None

    with open(Path(map_file), "r") as map_fp:
        in_summary_section = False

        for line in map_fp:
            if "MODULE SUMMARY" in line:
                in_summary_section = True
                continue

            if not in_summary_section:
                continue

            stripped = line.strip()

            if not stripped or stripped.startswith("+--"):
                continue

            parts = stripped.split()

            if len(parts) >= 1 and (parts[0].endswith(".a") or parts[0].endswith(".a/")):
                in_archive = parts[0].replace("./", "").strip()
                continue

            if stripped.startswith("Total:"):
                if len(parts) >= 4 and in_archive:
                    try:
                        code = int(parts[1])
                        ro_data = int(parts[2])
                        rw_data = int(parts[3])
                        archive_totals[in_archive] = [code, ro_data, rw_data]
                    except (ValueError, IndexError):
                        pass
                continue

            if len(parts) >= 3:
                try:
                    code = int(parts[-3])
                    ro_data = int(parts[-2])
                    rw_data = int(parts[-1])

                    file_name = " ".join(parts[:-3])
                    file_name = file_name.replace("./", "").strip()

                    if file_name and (file_name.endswith(".o") or file_name.endswith(".obj")):
                        memory_map[file_name] = [code, ro_data, rw_data]
                except (ValueError, IndexError):
                    pass

    return memory_map, archive_totals

def get_memory_mspm0(modelmaker_run, device, ARM_LLVM_CGT_PATH):
    fel_memory_dir = setup_paths()
    fel_memory_mspm0 = fel_memory_dir / "mspm0"

    files = [
      fel_memory_dir / "main_mspm0.c",
      fel_memory_mspm0 / "feature_extract.c",
    ]

    ti_cgt_dir = Path(ARM_LLVM_CGT_PATH)
    ti_cgt_include = ti_cgt_dir / "include"

    modelmaker_run_path = Path(modelmaker_run)
    modelmaker_compilation = modelmaker_run_path / "compilation" / "artifacts"
    modelmaker_quantization = modelmaker_run_path / "training" / "quantization" / "golden_vectors"

    compile_cmd = [
      f'{ti_cgt_dir / "bin" / "tiarmclang"}',
      "-c",
      "-mcpu=cortex-m0plus",
      "-mfloat-abi=soft",
      "-march=thumbv6",
      "-mlittle-endian",
      "-mthumb",
      "-Og",
      "-march=armv8.1-m.main+cdecp0",
      "-w",
      f'-I"{ti_cgt_include}"',
      f'-I"{modelmaker_compilation}"',
      f'-I"{modelmaker_quantization}"',
      f'-I"{fel_memory_mspm0}"',
    #   " > /dev/null 2>&1",
    ]

    for file_ in files:
      cmd = " ".join(compile_cmd) + " " + str(file_)
      os.system(cmd)

    ti_cgt_lib = ti_cgt_dir / "lib"
    fel_a = fel_memory_mspm0 / "fel.a"

    lnk_cmd = [
      f'{ti_cgt_dir / "bin" / "tiarmclang"}',
      "-mcpu=cortex-m0plus",
      "-mfloat-abi=soft",
      "-march=thumbv6",
      "-mlittle-endian",
      "-mthumb",
      "-Og",
      "-march=armv8.1-m.main+cdecp0",
      "-w",
      '-Wl,-m"app.map"',
      f'-Wl,-i"{ti_cgt_lib}"',
      f'-Wl,-i"{fel_memory_mspm0}"',
      "-Wl,--reread_libs",
      "-Wl,--rom_model",
      './main_mspm0.o',
      './feature_extract.o',
      f'-Wl,{fel_a}',
      "-Wl,-llibc.a",
    #   " > /dev/null 2>&1",
    ]

    cmd = " ".join(lnk_cmd)
    os.system(cmd)

    memory_map, archive_totals = get_memory("app.map")

    files_to_report = [
        ('main_mspm0.o', 'main_mspm0.o'),
        ('feature_extract.o', 'feature_extract.o'),
        (str(fel_a), 'fel.a'),
    ]

    print_memory_report(memory_map, archive_totals, files_to_report)

    cleanup_artifacts(['*.xml', 'app.map', '*.out', '*.o'])
    logger.info("TI memory calculation completed successfully")


def get_memory_am13(modelmaker_run, device, ARM_LLVM_CGT_PATH):
    fel_memory_dir = setup_paths()
    fel_memory_am13 = fel_memory_dir / "am13"

    files = [
      fel_memory_dir / "main.c",
      fel_memory_am13 / "feature_extract.c",
      fel_memory_am13 / "feature_extract_am13.c",
    ]

    ti_cgt_dir = Path(ARM_LLVM_CGT_PATH)
    ti_cgt_include = ti_cgt_dir / "include"

    modelmaker_run_path = Path(modelmaker_run)
    modelmaker_compilation = modelmaker_run_path / "compilation" / "artifacts"
    modelmaker_quantization = modelmaker_run_path / "training" / "quantization" / "golden_vectors"

    compile_cmd = [
      f'{ti_cgt_dir / "bin" / "tiarmclang"}',
      "-c",
      "-mcpu=cortex-m33",
      "-mfloat-abi=hard",
      "-mfpu=fpv5-sp-d16",
      "-mlittle-endian",
      "-mthumb",
      "-Og",
      "-march=armv8.1-m.main+cdecp0",
      "-w",
      f'-I"{ti_cgt_include}"',
      f'-I"{modelmaker_compilation}"',
      f'-I"{modelmaker_quantization}"',
      f'-I"{fel_memory_am13}"',
    #   " > /dev/null 2>&1",
    ]

    for file_ in files:
      cmd = " ".join(compile_cmd) + " " + str(file_)
      os.system(cmd)

    ti_cgt_lib = ti_cgt_dir / "lib"
    fel_a = fel_memory_am13 / "fel.a"

    lnk_cmd = [
      f'{ti_cgt_dir / "bin" / "tiarmclang"}',
      "-mcpu=cortex-m33",
      "-mfloat-abi=hard",
      "-mfpu=fpv5-sp-d16",
      "-mlittle-endian",
      "-mthumb",
      "-Og",
      "-march=armv8.1-m.main+cdecp0",
      '-Wl,-m"app.map"',
      f'-Wl,-i"{ti_cgt_lib}"',
      "-w",
      "-Wl,--reread_libs",
      "-Wl,--rom_model",
      './main.o',
      './feature_extract.o',
      './feature_extract_am13.o',
      f"-Wl,{fel_a}",
    #   " > /dev/null 2>&1",
    ]

    cmd = " ".join(lnk_cmd)
    os.system(cmd)

    memory_map, archive_totals = get_memory("app.map")

    files_to_report = [
        ('main.o', 'main.o'),
        ('feature_extract.o', 'feature_extract.o'),
        ('feature_extract_am13.o', 'feature_extract_am13.o'),
        (str(fel_a), 'fel.a'),
    ]

    print_memory_report(memory_map, archive_totals, files_to_report)

    cleanup_artifacts(['*.xml', 'app.map', '*.out', '*.o'])
    logger.info("TI memory calculation completed successfully")


def get_memory_c28(modelmaker_run, device, C2000_CG_ROOT):
    fel_memory_dir = setup_paths()

    fel_memory_c28 = fel_memory_dir / "c28"

    files = [
      fel_memory_dir / "main.c",
      fel_memory_c28 / "feature_extract.c",
      fel_memory_c28 / "feature_extract_c28.c",
    ]

    ti_cgt_c2000_dir = Path(C2000_CG_ROOT)
    ti_cgt_c2000_include = ti_cgt_c2000_dir / "include"
    fel_memory_c28_fpu = fel_memory_c28 / "FPU"
    fel_memory_c28_device = fel_memory_c28 / "driverlib" / device

    modelmaker_run_path = Path(modelmaker_run)
    modelmaker_compilation = modelmaker_run_path / "compilation" / "artifacts"
    modelmaker_quantization = modelmaker_run_path / "training" / "quantization" / "golden_vectors"

    compile_cmd = [
        f'{ti_cgt_c2000_dir / "bin" / "cl2000"}',
        "-v28",
        "-ml",
        "-mt",
        "--cla_support=cla2",
        "-DCPU1",
        "--float_support=fpu32",
        "--tmu_support=tmu0",
        "--vcu_support=vcrc",
        "--fp_mode=relaxed",
        "-Ooff",
        f'--include_path="{fel_memory_dir}"',
        f'--include_path="{fel_memory_c28}"',
        f'--include_path="{fel_memory_c28_fpu}"',
        f'--include_path="{fel_memory_c28_device}"',
        f'--include_path="{modelmaker_compilation}"',
        f'--include_path="{modelmaker_quantization}"',
        f'--include_path="{ti_cgt_c2000_include}"',
        f"--define=_LAUNCHXL_{device.upper()}",
        "--abi=eabi",
        " > /dev/null 2>&1",
    ]
    for file_ in files:
      cmd = " ".join(compile_cmd) + " " + str(file_)
      os.system(cmd)

    ti_cgt_c2000_lib = ti_cgt_c2000_dir / "lib"
    fel_a = fel_memory_c28 / "fel.a"

    lnk_cmd = [
        f'{ti_cgt_c2000_dir / "bin" / "cl2000"}',
        "-v28",
        "-ml",
        "-mt",
        "--cla_support=cla2",
        "-DCPU1",
        "--float_support=fpu32",
        "--tmu_support=tmu0",
        "--vcu_support=vcrc",
        "--fp_mode=relaxed",
        "-Ooff",
        f'--include_path="{fel_memory_dir}"',
        f'--include_path="{fel_memory_c28}"',
        f'--include_path="{fel_memory_c28_fpu}"',
        f'--include_path="{modelmaker_compilation}"',
        f'--include_path="{modelmaker_quantization}"',
        f'--include_path="{ti_cgt_c2000_include}"',
        f'--include_path="{ti_cgt_c2000_lib}"',
        f"--define=_LAUNCHXL_{device.upper()}",
        "--gen_func_subsections=on",
        "--abi=eabi",
        '-z -m"app.map"',
        "--ram_model",
        "./main.obj",
        "./feature_extract.obj",
        "./feature_extract_c28.obj",
        "-llibc.a",
        str(fel_a),
        " > /dev/null 2>&1",
    ]

    cmd = " ".join(lnk_cmd)
    os.system(cmd)

    memory_map, archive_totals = get_memory("app.map")

    files_to_report = [
        ('main.obj', 'main.obj'),
        ('feature_extract.obj', 'feature_extract.obj'),
        ('feature_extract_c28.obj', 'feature_extract_c28.obj'),
        (str(fel_a), 'fel.a'),
    ]
    print_memory_report(memory_map, archive_totals, files_to_report)

    cleanup_artifacts(['*.xml', 'app.map', '*.out', '*.obj'])
    logger.info("TI memory calculation completed successfully")


def get_memory_c29(modelmaker_run, device, CG_TOOL_ROOT):
    fel_memory_dir = setup_paths()

    fel_memory_c29 = fel_memory_dir / "c29"

    files = [
      fel_memory_dir / "main.c",
      fel_memory_c29 / "feature_extract.c",
      fel_memory_c29 / "feature_extract_c29.c",
    ]

    fel_memory_c29_driverlib = fel_memory_c29 / "driverlib"
    ti_cgt_c29_dir = Path(CG_TOOL_ROOT)
    ti_cgt_c29_include = ti_cgt_c29_dir / "include"
    
    modelmaker_run_path = Path(modelmaker_run)
    modelmaker_compilation = modelmaker_run_path / "compilation" / "artifacts"
    modelmaker_quantization = modelmaker_run_path / "training" / "quantization" / "golden_vectors"

    compile_cmd = [
        f'{ti_cgt_c29_dir / "bin" / "c29clang"}',
        "-c",
        "-O1",
        f'-I"{fel_memory_c29}"',
        f'-I"{modelmaker_compilation}"',
        f'-I"{modelmaker_quantization}"',
        f'-I"{ti_cgt_c29_include}"',
        "-g",
        "-w",
        " > /dev/null 2>&1",
    ]
    for file_ in files:
      cmd = " ".join(compile_cmd) + " " + str(file_)
      os.system(cmd)

    c29_fel_lib = fel_memory_c29 / "fel.a"

    lnk_cmd = [
        f'{ti_cgt_c29_dir / "bin" / "c29clang"}',
        "-O1",
        "-g",
        "-w",
        '-Wl,-m"app.map"',
        f'-Wl,-i"{fel_memory_dir}"',
        f'-Wl,-i"{fel_memory_c29}"',
        "-Wl,--reread_libs",
        "-Wl,--rom_model",
        "./main.o",
        "./feature_extract.o",
        "./feature_extract_c29.o",
        f"{c29_fel_lib}",
        " > /dev/null 2>&1",
    ]

    cmd = " ".join(lnk_cmd)
    os.system(cmd)

    memory_map, archive_totals = get_memory("app.map")

    files_to_report = [
        ('main.o', 'main.o'),
        ('feature_extract.o', 'feature_extract.o'),
        ('feature_extract_c29.o', 'feature_extract_c29.o'),
        (str(c29_fel_lib), 'fel.a'),
    ]
    print_memory_report(memory_map, archive_totals, files_to_report)

    cleanup_artifacts(['*.xml', 'app.map', '*.out', '*.o'])
    logger.info("TI memory calculation completed successfully")


def get_args_parser():
    DESCRIPTION = "Given user_input_config, this script generates .out, .map file for understanding the memory consumption of feature extraction"
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=str, required=True, help='Path of the run directory')
    parser.add_argument('--func-name', type=str, required=True, help="Function name for the core, eg: get_memory_c28, get_memory_mspm0")
    parser.add_argument('--compiler-path', type=str, required=True, help="Path of the compiler of the device")
    parser.add_argument('--device-name', type=str, required=True, help="Name of the device")
    return parser


def main(run_dir, config_func_name, config_tool_path, config_device_name):

    func = config_func_name
    tool_path = Path(config_tool_path)

    if not tool_path.exists():
        logger.error(f"Compiler tool path not found: {tool_path}")
        return 1

    if func == 'get_memory_c28':
        get_memory_c28(run_dir, config_device_name, tool_path)
    elif func == 'get_memory_c29':
        get_memory_c29(run_dir, config_device_name, tool_path)
    elif func == 'get_memory_am13':
        get_memory_am13(run_dir, config_device_name, tool_path)
    elif func == 'get_memory_mspm0':
        get_memory_mspm0(run_dir, config_device_name, tool_path)
    else:
        return 1
    return 0

def run(run_dir, config_func_name, config_tool_path, config_device_name):
    return main(run_dir, config_func_name, config_tool_path, config_device_name)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args.run_dir, args.func_name, args.compiler_path, args.device_name)