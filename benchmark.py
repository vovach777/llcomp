import os
import subprocess
import glob
import shutil

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

# Пути
COMPRESSOR = "build/llrice-c"
DECOMPRESSOR = "build/llrice-d"
SEARCH_PATHS = ["benchmark_work/professional_train_2020/*.pnm"]
NUT_DIR = "benchmark_work/professional_train_2020"

print(f"| {'File':<45} | {'Original (PPM)':<15} | {'Gemini (LLR)':<15} | {'Reference (NUT)':<15} | {'Ratio (LLR/Orig)':<18} | {'Status':<10} |")
print(f"|{'-'*47}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*20}|{'-'*12}|")

files = []
for path in SEARCH_PATHS:
    files.extend(glob.glob(path))
files.sort()

total_ppm = 0
total_llr = 0
total_nut = 0
nut_count = 0


for ppm_file in files:
    base_name = os.path.basename(ppm_file)
    # Если это уже результат распаковки (.ppm.llr.ppm), пропускаем
    if ".llr" in base_name:
        continue
        
    file_root = os.path.splitext(base_name)[0]
    
    llr_file = ppm_file + ".llr"
    # Декомпрессор сохраняет как <input_filename>.ppm
    expected_output = llr_file + ".ppm"
    
    # Очистка перед тестом
    if os.path.exists(llr_file): os.remove(llr_file)
    if os.path.exists(expected_output): os.remove(expected_output)
    
    # 1. Сжатие
    try:
        res = subprocess.run([COMPRESSOR, ppm_file, llr_file], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"| {base_name:<45} | {'ERROR (C)':<15} | {'-':<15} | {'-':<15} | {'-':<18} | {str(res.returncode):<10} |")
            continue
    except Exception as e:
        print(f"| {base_name:<45} | {'EXCEPT (C)':<15} | {str(e):<15} |")
        continue

    # 2. Распаковка (аргумент только один!)
    try:
        res = subprocess.run([DECOMPRESSOR, llr_file], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"| {base_name:<45} | {'ERROR (D)':<15} | {'-':<15} | {'-':<15} | {'-':<18} | {str(res.returncode):<10} |")
            continue
    except Exception as e:
        print(f"| {base_name:<45} | {'EXCEPT (D)':<15} | {str(e):<15} |")
        continue

    # 3. Проверка идентичности
    if not os.path.exists(expected_output):
         print(f"| {base_name:<45} | {'MISSING':<15} | {'-':<15} | {'-':<15} | {'-':<18} | {'Fail':<10} |")
         continue

    status = compare_ppm(ppm_file, expected_output)

    # 4. Размеры
    size_ppm = os.path.getsize(ppm_file)
    size_llr = os.path.getsize(llr_file)
    
    # Поиск NUT файла
    nut_candidate = os.path.join(NUT_DIR, file_root + ".nut")
    
    if os.path.exists(nut_candidate):
        size_nut = os.path.getsize(nut_candidate)
        nut_str = f"{size_nut:,}"
    else:
        nut_str = "N/A"

    ratio = (size_llr / size_ppm) * 100
    
    print(f"| {base_name:<45} | {size_ppm:<15,} | {size_llr:<15,} | {nut_str:<15} | {ratio:<17.2f}% | {status:<10} |")

    total_ppm += size_ppm
    total_llr += size_llr

    if os.path.exists(nut_candidate):
       total_nut += size_nut
       nut_count += 1

    # Чистка (удаляем распакованный файл)
    os.remove(expected_output)

print(f"|{'='*47}|{'='*17}|{'='*17}|{'='*17}|{'='*20}|{'='*12}|")

total_ratio = (total_llr / total_ppm) * 100 if total_ppm else 0
nut_total_str = f"{total_nut:,}" if nut_count else "N/A"

print(
    f"| {'TOTAL':<45} | "
    f"{total_ppm:<15,} | "
    f"{total_llr:<15,} | "
    f"{nut_total_str:<15} | "
    f"{total_ratio:<17.2f}% | "
    f"{'':<10} |"
)
