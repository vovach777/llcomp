#!/usr/bin/env python3
"""Regression tests for the C++ (master) llcomp engine.

Usage:
    python3 tests/roundtrip.py --build build/
    ctest --test-dir build/ --output-on-failure

Covers:
  * byte-exact round-trip of P6 (RGB) and P5 (grayscale) for all three models;
  * graceful, non-crashing errors on truncated PPM and malformed/truncated .llc.
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

MODELS = (("standard", "", ""), ("small", "-s", "-s"), ("large", "-xl", "-xl"))


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def make_fixtures(root, w=64, h=48):
    gray = bytes((x * 3 + y * 5 + (x * y) // 7) & 255 for y in range(h) for x in range(w))
    color = bytes((x * 7 + y * 3 + k * 40) & 255
                  for y in range(h) for x in range(w) for k in range(3))
    root.joinpath("small.pgm").write_bytes(b"P5\n%d %d\n255\n" % (w, h) + gray)
    root.joinpath("small.ppm").write_bytes(b"P6\n%d %d\n255\n" % (w, h) + color)
    root.joinpath("truncated.pgm").write_bytes(b"P5\n%d %d\n255\n" % (w, h) + gray[:100])
    root.joinpath("p4.pbm").write_bytes(b"P4\n%d %d\n255\n" % (w, h) + gray[:100])
    root.joinpath("maxval16.pgm").write_bytes(b"P5\n%d %d\n65535\n" % (w, h) + gray)
    header = b"\x7b\x01" + struct.pack("<HH", w, h)
    root.joinpath("empty.llc").write_bytes(b"")
    root.joinpath("short2.llc").write_bytes(b"\x7b\x01")
    root.joinpath("short6.llc").write_bytes(header)
    root.joinpath("badch.llc").write_bytes(b"\x7b\x00" + struct.pack("<HH", w, h) + bytes(64))
    root.joinpath("badmagic.llc").write_bytes(b"\x00\x01" + struct.pack("<HH", w, h) + bytes(64))


def expect_error(proc, label, errors):
    out = proc.stdout.decode("utf-8", "replace")
    bad = any(s in out for s in ("AddressSanitizer", "runtime error", "terminate called",
                                 "Segmentation", "double free"))
    if proc.returncode != 1 or bad or "Error" not in out:
        errors.append("%s: rc=%d, sanitizer/crash=%s, output=%r"
                      % (label, proc.returncode, bad, out.strip()[:160]))
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="CMake build directory with llcompc/llcompd")
    build = Path(ap.parse_args().build).resolve()
    errors, passed = [], 0

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_fixtures(root)
        for label, cs, ds in MODELS:
            comp, decomp = build / ("llcompc" + cs), build / ("llcompd" + ds)
            if not comp.exists() or not decomp.exists():
                errors.append("%s: binaries not found in %s" % (label, build))
                continue
            for ext in ("ppm", "pgm"):
                src, enc, out = root / ("small." + ext), root / "t.llc", root / ("out." + ext)
                a = run([str(comp), str(src), str(enc)])
                b = run([str(decomp), str(enc), str(out)]) if a.returncode == 0 else None
                ok = (a.returncode == 0 and b is not None and b.returncode == 0
                      and out.exists() and out.read_bytes() == src.read_bytes())
                if ok:
                    passed += 1
                    print("PASS  %-8s %-3s byte-exact (%d B)" % (label, ext, enc.stat().st_size))
                else:
                    errors.append("%s %s: round-trip failed (enc rc=%s)"
                                  % (label, ext, a.returncode))
                    print("FAIL  %-8s %-3s" % (label, ext))

        comp, decomp = build / "llcompc", build / "llcompd"
        negatives = [
            ("truncated PPM", [str(comp), str(root / "truncated.pgm"), str(root / "n.llc")]),
            ("P4 input", [str(comp), str(root / "p4.pbm"), str(root / "n.llc")]),
            ("maxvalue 65535", [str(comp), str(root / "maxval16.pgm"), str(root / "n.llc")]),
        ]
        for name in ("empty.llc", "short2.llc", "short6.llc", "badch.llc", "badmagic.llc"):
            negatives.append((name, [str(decomp), str(root / name), str(root / "n.pgm")]))
        for label, cmd in negatives:
            if expect_error(run(cmd), label, errors):
                passed += 1
                print("PASS  %-16s graceful error" % label)
            else:
                print("FAIL  %-16s" % label)

    print("\n%d passed, %d failed" % (passed, len(errors)))
    for e in errors:
        print("  - " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
