import os
import subprocess
import glob
import shutil
import time
import sys
import random
import json
import shlex

def split_cmd(cmd):
    """'build/x.exe -t 2 -s 2' -> ('build/x.exe', ['-t', '2', '-s', '2']); quotes allowed for paths with spaces."""
    parts = [p.strip('"') for p in shlex.split(cmd, posix=False)]
    return parts[0], parts[1:]

def read_ppm_p6(filename):
    """Reads a P6 PPM file, returns header info and pixel data, handling comments."""
    try:
        with open(filename, 'rb') as f:
            magic_number = f.readline().strip()
            if magic_number != b'P6':
                return None, None, f"Not a P6 file: {magic_number.decode()}"

            header_values = []
            while len(header_values) < 3:
                line = f.readline()
                if line.startswith(b'#'):
                    continue
                header_values.extend(line.split())

            try:
                width = int(header_values[0])
                height = int(header_values[1])
                max_val = int(header_values[2])
            except (ValueError, IndexError):
                 return None, None, "Invalid PPM header"

            pixel_data = f.read()
            header_info = {'width': width, 'height': height, 'max_val': max_val}
            return header_info, pixel_data, "OK"
    except IOError as e:
        return None, None, f"IOError: {e}"

def get_ppm_header(filename):
    """Reads only the header of a P6 PPM file to get dimensions."""
    try:
        with open(filename, 'rb') as f:
            magic_number = f.readline().strip()
            if magic_number != b'P6':
                return None

            header_values = []
            while len(header_values) < 3:
                line = f.readline()
                if line.startswith(b'#'):
                    continue
                header_values.extend(line.split())

            try:
                width = int(header_values[0])
                height = int(header_values[1])
                max_val = int(header_values[2])
                return {'width': width, 'height': height, 'max_val': max_val}
            except:
                return None
    except:
        return None

def compare_ppm(file1, file2):
    """Compares two PPM files, ignoring comments."""
    header1, data1, status1 = read_ppm_p6(file1)
    if status1 != "OK": return f"ERR_READ1: {status1}"

    header2, data2, status2 = read_ppm_p6(file2)
    if status2 != "OK": return f"ERR_READ2: {status2}"

    if header1['width'] != header2['width'] or header1['height'] != header2['height']: return "DIFF_RES"
    if header1['max_val'] != header2['max_val']: return "DIFF_MAXVAL"
    if data1 != data2: return "DIFF_PIXELS"

    return "OK"

# --- Парсинг аргументов командной строки ---
time_limit = None
shuffle_files = False
file_limit = None
config_path = "config.json"

for arg in sys.argv:
    if arg.startswith("-t"):
        try: time_limit = int(arg[2:])
        except ValueError: print(f"Invalid time limit: {arg}")
    elif arg.startswith("-n"):
        try: file_limit = int(arg[2:])
        except ValueError: print(f"Invalid file limit: {arg}")
    elif arg == "-p":
        shuffle_files = True
    elif arg.endswith(".json"):
        config_path = arg

# --- Загрузка конфигурации ---
if not os.path.exists(config_path):
    print(f"Configuration file '{config_path}' not found!")
    sys.exit(1)

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

bench_progs = config.get("bench_prog", [])
datasets = config.get("dataset", [])
bench_dir = config.get("bench_dir", "./bench_workdir")

# --- Подготовка рабочей директории ---
os.makedirs(bench_dir, exist_ok=True)

# Копируем кодеки в RAM-диск и обновляем пути
for prog in bench_progs:
    # "enc"/"dec" may carry arguments: "build/llcompc-rc.exe -t 2 -s 2"
    enc_exe, prog['enc_args'] = split_cmd(prog['enc'])
    dec_exe, prog['dec_args'] = split_cmd(prog['dec'])

    prog['ram_enc'] = os.path.join(bench_dir, os.path.basename(enc_exe))
    prog['ram_dec'] = os.path.join(bench_dir, os.path.basename(dec_exe))

    shutil.copy(enc_exe, prog['ram_enc'])
    shutil.copy(dec_exe, prog['ram_dec'])
    
    # Инициализация статистики
    prog['stats'] = {
        'enc_time': 0.0, 'dec_time': 0.0,
        'total_size': 0, 'total_pixels': 0,
        'processed_count': 0
    }

# --- Поиск файлов ---
files = []
for path in datasets:
    files.extend(glob.glob(os.path.join(path, "*.pnm")))
    files.extend(glob.glob(os.path.join(path, "*.ppm")))

if shuffle_files: random.shuffle(files)
else: files.sort()

if file_limit is not None: files = files[:file_limit]

# --- Отрисовка шапки таблицы 1 ---
header_str = f"| {'File':<35} | {'Original':<15} |"
sep_str =    f"|{'-'*37}|{'-'*17}|"
for prog in bench_progs:
    header_str += f" {prog['name']:<15} | {'BPP':<6} |"
    sep_str +=    f"{'-'*17}|{'-'*8}|"
header_str += f" {'Status':<15} |"
sep_str +=    f"{'-'*17}|"

print(header_str)
print(sep_str)

total_ppm_size = 0
total_ppm_pixels = 0
processed_total = 0
start_time = time.time()

# --- Главный цикл бенчмарка ---
for orig_file in files:
    if time_limit is not None and (time.time() - start_time) > time_limit:
        break

    base_name = os.path.basename(orig_file)
    display_name = base_name[:35]
    
    # Копируем картинку в RAM-диск
    work_img = os.path.join(bench_dir, base_name)
    shutil.copy(orig_file, work_img)

    meta = get_ppm_header(work_img)
    num_pixels = meta['width'] * meta['height'] if meta else 0
    orig_size = os.path.getsize(work_img)

    if num_pixels == 0:
        print(f"| {display_name:<35} | {'ERR_HEAD':<15} | ...")
        os.remove(work_img)
        continue

    total_ppm_size += orig_size
    total_ppm_pixels += num_pixels
    processed_total += 1

    row_out = f"| {display_name:<35} | {orig_size:<15,} |"
    statuses = []

    # Прогоняем каждый кодек
    for prog in bench_progs:
        ext = prog['ext']
        comp_file = work_img + "." + ext
        decomp_file = comp_file + ".pnm"
        
        codec_status = "OK"
        comp_size = 0
        bpp = 0.0

        # Сжатие
        try:
            t0 = time.time()
            res = subprocess.run([prog['ram_enc'], *prog['enc_args'], work_img, comp_file], capture_output=True, text=True)
            t1 = time.time()
            if res.returncode != 0:
                codec_status = f"ERR_C({res.returncode})"
            else:
                prog['stats']['enc_time'] += (t1 - t0)
        except Exception as e:
            codec_status = "EXC_C"

        # Распаковка (только если сжатие прошло успешно)
        if codec_status == "OK":
            try:
                t0 = time.time()
                res = subprocess.run([prog['ram_dec'], *prog['dec_args'], comp_file, decomp_file], capture_output=True, text=True)
                t1 = time.time()
                if res.returncode != 0:
                    codec_status = f"ERR_D({res.returncode})"
                else:
                    prog['stats']['dec_time'] += (t1 - t0)
            except Exception as e:
                codec_status = "EXC_D"

        # Валидация и сбор метрик
        if codec_status == "OK":
            if os.path.exists(decomp_file):
                codec_status = compare_ppm(work_img, decomp_file)
            else:
                codec_status = "MISSING"

        if codec_status == "OK" and os.path.exists(comp_file):
            comp_size = os.path.getsize(comp_file)
            bpp = (comp_size * 8) / num_pixels
            prog['stats']['total_size'] += comp_size
            prog['stats']['total_pixels'] += num_pixels
            prog['stats']['processed_count'] += 1
            
            row_out += f" {comp_size:<15,} | {bpp:<6.2f} |"
        else:
            row_out += f" {'-':<15} | {'-':<6} |"
            statuses.append(f"{prog['name']}:{codec_status}")

        # Очистка артефактов кодека
        if os.path.exists(comp_file): os.remove(comp_file)
        if os.path.exists(decomp_file): os.remove(decomp_file)

    # Итоговый статус по файлу
    final_status = "OK" if not statuses else ",".join(statuses)
    row_out += f" {final_status:<15} |"
    print(row_out)

    # Удаляем скопированный оригинал из RAM-диска
    os.remove(work_img)

print(sep_str)

# --- Итоговая строка BPP ---
total_row = f"| {'TOTAL':<35} | {total_ppm_size:<15,} |"
for prog in bench_progs:
    st = prog['stats']
    tot_size = st['total_size']
    tot_bpp = (tot_size * 8) / st['total_pixels'] if st['total_pixels'] > 0 else 0
    size_str = f"{tot_size:,}" if tot_size > 0 else "N/A"
    bpp_str = f"{tot_bpp:.2f}" if tot_size > 0 else "-"
    
    total_row += f" {size_str:<15} | {bpp_str:<6} |"
total_row += f" {'':<15} |"
print(total_row)

total_mb = total_ppm_size / (1024 * 1024)

# --- Таблица 2: Скорость кодирования ---
print(f"\nEncoding Performance ({processed_total} files, {total_mb:.2f} MB total):")
print(f"| {'Codec':<15} | {'Total Time (s)':<15} | {'Speed (MB/s)':<15} |")
print(f"|{'-'*17}|{'-'*17}|{'-'*17}|")
for prog in bench_progs:
    enc_time = prog['stats']['enc_time']
    # Считаем скорость только по успешно обработанным файлам (приблизительно)
    if enc_time > 0:
        speed = total_mb / enc_time
        print(f"| {prog['name']:<15} | {enc_time:<15.4f} | {speed:<15.2f} |")
    else:
        print(f"| {prog['name']:<15} | {'0.0000':<15} | {'N/A':<15} |")

# --- Таблица 3: Скорость декодирования ---
print(f"\nDecoding Performance:")
print(f"| {'Codec':<15} | {'Total Time (s)':<15} | {'Speed (MB/s)':<15} |")
print(f"|{'-'*17}|{'-'*17}|{'-'*17}|")
for prog in bench_progs:
    dec_time = prog['stats']['dec_time']
    if dec_time > 0:
        speed = total_mb / dec_time
        print(f"| {prog['name']:<15} | {dec_time:<15.4f} | {speed:<15.2f} |")
    else:
        print(f"| {prog['name']:<15} | {'0.0000':<15} | {'N/A':<15} |")