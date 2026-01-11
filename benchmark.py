import os
import subprocess
import glob
import filecmp
import shutil

# Пути
COMPRESSOR = "build/llrice-c"
DECOMPRESSOR = "build/llrice-d"
SEARCH_PATHS = ["samples/ppm/*.ppm", "samples/test_nut_extracted/*.ppm"]
NUT_DIR = "build"

print(f"| {'File':<45} | {'Original (PPM)':<15} | {'Gemini (LLR)':<15} | {'Reference (NUT)':<15} | {'Ratio (LLR/Orig)':<18} | {'Status':<10} |")
print(f"|{'-'*46}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*20}|{'-'*12}|")

files = []
for path in SEARCH_PATHS:
    files.extend(glob.glob(path))
files.sort()

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

    if filecmp.cmp(ppm_file, expected_output, shallow=False):
        status = "OK"
    else:
        status = "DIFF!"

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

    # Чистка (удаляем распакованный файл)
    os.remove(expected_output)
