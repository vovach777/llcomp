import os
import subprocess
import glob
import shutil
import time
import sys

def read_ppm_p6(filename):
    """Reads a P6 PPM file, returns header info and pixel data, handling comments."""
    try:
        with open(filename, 'rb') as f:
            # 1. Read Magic Number
            magic_number = f.readline().strip()
            if magic_number != b'P6':
                return None, None, f"Not a P6 file: {magic_number.decode()}"

            # 2. Read Width, Height, Maxval, skipping comments
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

            # 3. Read Data
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
    if status1 != "OK":
        return f"ERR_READ1: {status1}"

    header2, data2, status2 = read_ppm_p6(file2)
    if status2 != "OK":
        return f"ERR_READ2: {status2}"

    if header1['width'] != header2['width'] or header1['height'] != header2['height']:
        return "DIFF_RES"

    if header1['max_val'] != header2['max_val']:
        return "DIFF_MAXVAL"

    if data1 != data2:
        return "DIFF_PIXELS"

    return "OK"

# Parse arguments
time_limit = None
for arg in sys.argv:
    if arg.startswith("-t"):
        try:
            time_limit = int(arg[2:])
            print(f"Time limit set to: {time_limit} seconds")
        except ValueError:
            print(f"Invalid time limit format: {arg}")

# Пути
COMPRESSOR = "build/llrice-c"
DECOMPRESSOR = "build/llrice-d"
SEARCH_PATHS = ["M:/professional_train_2020/*.pnm"]
NUT_DIR = "M:/professional_train_2020"
HALIC_DIR = "M:/professional_train_2020"

print(f"| {'File':<35} | {'Original':<15} | {'LLR':<15} | {'BPP':<6} | {'FFV1':<15} | {'BPP':<6} | {'HALIC':<15} | {'BPP':<6} | {'Status':<10} |")
print(f"|{'-'*37}|{'-'*17}|{'-'*17}|{'-'*8}|{'-'*17}|{'-'*8}|{'-'*17}|{'-'*8}|{'-'*12}|")

files = []
for path in SEARCH_PATHS:
    files.extend(glob.glob(path))
files.sort()

total_ppm = 0
total_llr = 0
total_nut = 0
total_halic = 0
total_pixels = 0
total_pixels_nut = 0
total_pixels_halic = 0

nut_count = 0
halic_count = 0
processed_count = 0

total_enc_time = 0.0
total_dec_time = 0.0

start_time = time.time()

for ppm_file in files:
    # Check time limit
    if time_limit is not None:
        if (time.time() - start_time) > time_limit:
            break

    base_name = os.path.basename(ppm_file)
    # Если это уже результат распаковки (.ppm.llr.ppm), пропускаем
    if ".llr" in base_name:
        continue

    # Обрезаем имя файла до 35 символов
    display_name = base_name[:35]

    file_root = os.path.splitext(base_name)[0]

    llr_file = ppm_file + ".llr"
    # Декомпрессор сохраняет как <input_filename>.ppm
    expected_output = llr_file + ".ppm"

    # Очистка перед тестом
    if os.path.exists(llr_file): os.remove(llr_file)
    if os.path.exists(expected_output): os.remove(expected_output)

    # Get metadata for BPP calc
    meta = get_ppm_header(ppm_file)
    if meta:
        num_pixels = meta['width'] * meta['height']
    else:
        num_pixels = 0
        print(f"| {display_name:<35} | {'ERR_HEAD':<15} | ...")
        continue

    # 1. Сжатие
    try:
        t0 = time.time()
        res = subprocess.run([COMPRESSOR, ppm_file, llr_file], capture_output=True, text=True)
        t1 = time.time()
        if res.returncode != 0:
            print(f"| {display_name:<35} | {'ERROR (C)':<15} | {'-':<15} | {'-':<6} | {'-':<15} | {'-':<6} | {'-':<15} | {'-':<6} | {str(res.returncode):<10} |")
            continue
        total_enc_time += (t1 - t0)
    except Exception as e:
        print(f"| {display_name:<35} | {'EXCEPT (C)':<15} | {str(e):<15} |")
        continue

    # 2. Распаковка (аргумент только один!)
    try:
        t0 = time.time()
        res = subprocess.run([DECOMPRESSOR, llr_file], capture_output=True, text=True)
        t1 = time.time()
        if res.returncode != 0:
            print(f"| {display_name:<35} | {'ERROR (D)':<15} | {'-':<15} | {'-':<6} | {'-':<15} | {'-':<6} | {'-':<15} | {'-':<6} | {str(res.returncode):<10} |")
            continue
        total_dec_time += (t1 - t0)
    except Exception as e:
        print(f"| {display_name:<35} | {'EXCEPT (D)':<15} | {str(e):<15} |")
        continue

    # 3. Проверка идентичности
    if not os.path.exists(expected_output):
         print(f"| {display_name:<35} | {'MISSING':<15} | {'-':<15} | {'-':<6} | {'-':<15} | {'-':<6} | {'-':<15} | {'-':<6} | {'Fail':<10} |")
         continue

    status = compare_ppm(ppm_file, expected_output)

    # 4. Размеры и BPP
    size_ppm = os.path.getsize(ppm_file)
    size_llr = os.path.getsize(llr_file)
    bpp_llr = (size_llr * 8) / num_pixels if num_pixels > 0 else 0

    # Поиск NUT файла
    nut_candidate = os.path.join(NUT_DIR, file_root + ".nut")

    if os.path.exists(nut_candidate):
        size_nut = os.path.getsize(nut_candidate)
        nut_str = f"{size_nut:,}"
        bpp_nut = (size_nut * 8) / num_pixels if num_pixels > 0 else 0
        nut_bpp_str = f"{bpp_nut:.2f}"
    else:
        size_nut = 0
        nut_str = "N/A"
        nut_bpp_str = "-"

    # Поиск HALIC файла
    halic_candidate = os.path.join(HALIC_DIR, file_root + ".halic")

    if os.path.exists(halic_candidate):
        size_halic = os.path.getsize(halic_candidate)
        halic_str = f"{size_halic:,}"
        bpp_halic = (size_halic * 8) / num_pixels if num_pixels > 0 else 0
        halic_bpp_str = f"{bpp_halic:.2f}"
    else:
        size_halic = 0
        halic_str = "N/A"
        halic_bpp_str = "-"

    print(f"| {display_name:<35} | {size_ppm:<15,} | {size_llr:<15,} | {bpp_llr:<6.2f} | {nut_str:<15} | {nut_bpp_str:<6} | {halic_str:<15} | {halic_bpp_str:<6} | {status:<10} |")

    total_ppm += size_ppm
    total_llr += size_llr
    total_pixels += num_pixels
    processed_count += 1

    if size_nut > 0:
       total_nut += size_nut
       nut_count += 1
       total_pixels_nut += num_pixels

    if size_halic > 0:
       total_halic += size_halic
       halic_count += 1
       total_pixels_halic += num_pixels

    # Чистка (удаляем распакованный файл)
    os.remove(expected_output)

print(f"|{'='*37}|{'='*17}|{'='*17}|{'='*8}|{'='*17}|{'='*8}|{'='*17}|{'='*8}|{'='*12}|")

nut_total_str = f"{total_nut:,}" if nut_count else "N/A"
halic_total_str = f"{total_halic:,}" if halic_count else "N/A"

# Calculate Total BPP
total_bpp_llr = (total_llr * 8) / total_pixels if total_pixels > 0 else 0
total_bpp_nut = (total_nut * 8) / total_pixels_nut if total_pixels_nut > 0 else 0
total_bpp_halic = (total_halic * 8) / total_pixels_halic if total_pixels_halic > 0 else 0

total_bpp_llr_str = f"{total_bpp_llr:.2f}" if total_pixels > 0 else "-"
total_bpp_nut_str = f"{total_bpp_nut:.2f}" if total_pixels_nut > 0 else "-"
total_bpp_halic_str = f"{total_bpp_halic:.2f}" if total_pixels_halic > 0 else "-"


print(
    f"| {'TOTAL':<35} | "
    f"{total_ppm:<15,} | "
    f"{total_llr:<15,} | "
    f"{total_bpp_llr_str:<6} | "
    f"{nut_total_str:<15} | "
    f"{total_bpp_nut_str:<6} | "
    f"{halic_total_str:<15} | "
    f"{total_bpp_halic_str:<6} | "
    f"{'':<10} |")

# Performance Table
print(f"\nPerformance Statistics ({processed_count} files):")
print(f"| {'Operation':<20} | {'Time (s)':<12} | {'Speed (MB/s)':<15} |")
print(f"|{'-'*22}|{'-'*14}|{'-'*17}|")

total_mb = total_ppm / (1024 * 1024)

if total_enc_time > 0:
    enc_speed = total_mb / total_enc_time
    print(f"| {'LLR Compression':<20} | {total_enc_time:<12.4f} | {enc_speed:<15.2f} |")
else:
    print(f"| {'LLR Compression':<20} | {'0.0000':<12} | {'N/A':<15} |")

if total_dec_time > 0:
    dec_speed = total_mb / total_dec_time
    print(f"| {'LLR Decompression':<20} | {total_dec_time:<12.4f} | {dec_speed:<15.2f} |")
else:
    print(f"| {'LLR Decompression':<20} | {'0.0000':<12} | {'N/A':<15} |")
print(f"|{'-'*22}|{'-'*14}|{'-'*17}|")
