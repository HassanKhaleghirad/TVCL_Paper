import io
import re
import json
import zipfile
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

DOWNLOADS = Path.home() / "Downloads"
STEM = "InTAS_highway_2_constantPositionOffset"
EXPECTED_MD5 = "4b8aa6b684a093de218256bae74c1a6e"

stats = defaultdict(Counter)
file_counts = Counter()
errors = []
json_examples = {}

def get_split(path):
    p = path.replace("\\", "/").lower()
    for split in ("train", "validation", "test"):
        if re.search(rf"/{split}/", "/" + p.strip("/") + "/"):
            return split.upper()
    return "UNKNOWN"

def process_json(raw, path):
    split = get_split(path)
    try:
        obj = json.loads(raw.decode("utf-8-sig"))
        records = obj if isinstance(obj, list) else [obj]
        file_counts[split] += 1
        stats[split]["records"] += len(records)

        for row in records:
            if not isinstance(row, dict):
                stats[split]["invalid_records"] += 1
                continue

            if "attacker" not in row:
                stats[split]["missing_label"] += 1
                continue

            label = row["attacker"]
            if isinstance(label, bool):
                stats[split]["invalid_label"] += 1
            elif label in (0, "0"):
                stats[split]["label_0"] += 1
            elif label in (1, "1"):
                stats[split]["label_1"] += 1
            else:
                stats[split]["invalid_label"] += 1

        json_examples.setdefault(split, path)
    except Exception as exc:
        errors.append((path, str(exc)))

def process_zip(raw, virtual_path, depth=0):
    if depth > 8:
        errors.append((virtual_path, "Nested ZIP depth exceeded"))
        return

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue

                path = virtual_path + "/" + item.filename
                low = item.filename.lower()

                try:
                    contents = z.read(item)
                    if low.endswith(".zip"):
                        process_zip(contents, path, depth + 1)
                    elif low.endswith(".json"):
                        process_json(contents, path)
                except Exception as exc:
                    errors.append((path, str(exc)))
    except Exception as exc:
        errors.append((virtual_path, str(exc)))

# Prefer the original archive: it should contain Train, Validation, and Test.
candidates = list(DOWNLOADS.glob(STEM + "*.zip"))
if not candidates:
    candidates = list(DOWNLOADS.rglob(STEM + "*.zip"))

# Prefer the outermost matching archive to avoid counting nested data twice.
candidates = sorted(
    set(candidates),
    key=lambda p: (len(p.parts), len(str(p)))
)

if candidates:
    archive = candidates[0]
    print("Archive selected:", archive)

    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    print("MD5:", digest)
    print("Official MD5 matches:", digest == EXPECTED_MD5)

    if digest != EXPECTED_MD5:
        print("WARNING: Verify that this is the official archive.")

    process_zip(archive.read_bytes(), archive.name)
else:
    print("No original ZIP found. Falling back to extracted JSON files.")
    roots = [
        p for p in DOWNLOADS.glob(STEM + "*")
        if p.is_dir()
    ]

    if not roots:
        raise SystemExit("Dataset folder not found in Downloads.")

    root = sorted(roots, key=lambda p: len(p.parts))[0]
    print("Dataset directory:", root)

    json_files = list(root.rglob("*.json"))
    for path in json_files:
        try:
            process_json(path.read_bytes(), str(path.relative_to(root)))
        except Exception as exc:
            errors.append((str(path), str(exc)))

print("\n" + "=" * 65)
print("LABEL REPORT BY SPLIT")
print("=" * 65)

for split in ("TRAIN", "VALIDATION", "TEST", "UNKNOWN"):
    s = stats[split]
    if not s:
        continue

    n0 = s["label_0"]
    n1 = s["label_1"]
    labeled = n0 + n1

    print(f"\n[{split}]")
    print("JSON files:", file_counts[split])
    print("Records:", s["records"])
    print("attacker=0:", n0)
    print("attacker=1:", n1)
    print("Missing labels:", s["missing_label"])
    print("Invalid labels:", s["invalid_label"])
    print("Invalid records:", s["invalid_records"])

    if labeled:
        print(f"Label 0 percentage: {100*n0/labeled:.2f}%")
        print(f"Label 1 percentage: {100*n1/labeled:.2f}%")

    if split in json_examples:
        print("Sample source:", json_examples[split])

if errors:
    print("\nErrors:", len(errors))
    for path, err in errors[:10]:
        print(path, "->", err)

print("\nCompleted.")
