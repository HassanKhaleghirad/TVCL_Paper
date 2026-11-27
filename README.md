# TVCL on InTAS (constantPositionOffset) - vehicle-disjoint protocol

pip install torch scikit-learn pandas numpy scipy
# all veh_*.json files in one folder (default /mnt/user-data/uploads; change with --data)

# 1) config selection on VALIDATION AUC only (test vehicles fixed, never used)
python tune.py
# 2) final run: fixed test vehicles, 20 seeds (inner train/val split + init vary)
python tvcl_intas.py --fixed-test --cfg "dict(skip=True,d=32,drop=0.3,wd=1e-3)" --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --epochs 200 --out detection_runs_intas.csv
# 3) ablation rows (Full / w/o MAR / w/o CSSL / w/o EAD)
python tvcl_intas.py --fixed-test --cfg "dict(skip=True,d=32,drop=0.3,wd=1e-3)" --ablation --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 200 --out ablation_runs_intas.csv

Output CSV = detection_runs.csv format for gen_tables.py (dataset key: intas).
Also written for the same split: GBM and LogReg on the same snapshot features (reference baselines).



python tvcl_onoff.py --dataset intas --data "C:\Users\hassan\TVCL_NEW\pythonProject1\dataset" --rebin 4 --diag-only

python tvcl_onoff.py --dataset intas --data "C:\Users\hassan\TVCL_NEW\pythonProject1\dataset" --rebin 4 --seeds 0 1 2 3 4 5 6 7 8 9 --out ablation_intas_rebin.csv


python tvcl_onoff.py --mode onoff   --seeds 0 1 2 3 4 --out runs_onoff.csv
python tvcl_onoff.py --mode patient --seeds 0 1 2 3 4 --out runs_patient.csv


python tvcl_intas.py --data "C:\Users\hassan\TVCL_NEW\pythonProject1\dataset" --ablation --seeds 0 1 2 3 4 5 6 7 8 9 --out ablation_intas_fixed.csv



python tvcl_real.py --data dataset --diag-only
python tvcl_real.py --data dataset --seeds 0 1 2 3 4 5 6 7 8 9 --out runs_real.csv
python tvcl_real.py --data dataset --seeds 0 1 2 3 4 5 6 7 8 9 --no-road --out runs_real_noroad.csv


این نسخه شامل موارد زیر است:
مؤلفه	پیاده‌سازی
MAR	Reliability-aware Multi-scale Attention
CSSL	Cross-view Contrastive Learning
EAD	Edge-aware Attention Aggregation
Temporal	تاریخچه علّی ۳ و ۲۰ پیام
Graph	ارتباط مکانی و زمانی خودروها
Split	تفکیک خودروها به‌صورت ۷۰/۱۵/۱۵
Ablation	حذف مستقل و ترکیبی ماژول‌ها

نکته مهم: این فایل یک پیاده‌سازی مستقل از معماری است، نه Patch تأییدشده روی tvcl_intas.py اصلی؛ چون آن فایل را هنوز ندارم.
۱. ساختار پوشه
در PyCharm ساختار زیر را داشته باشید:
pythonProject1/
│
├── tvcl_real.py
│
├── dataset/
│   ├── veh_0001.json
│   ├── veh_0002.json
│   ├── veh_0003.json
│   ├── ...
│   └── veh_0841.json
│
└── results/


کد تمام فایل‌های JSON را به‌صورت بازگشتی می‌خواند و رکوردهای تکراری را حذف می‌کند.
۲. نصب کتابخانه‌ها
در PowerShell محیط مجازی PyCharm:
pip install numpy pandas scikit-learn scipy torch


۳. اجرای آزمایش
ابتدا:
python tvcl_real.py `
  --data ".\dataset" `
  --epochs 100 `
  --patience 15 `
  --seeds 0 1 2 3 4 `
  --out ".\results\tvcl_real_ablation.csv"


برای آزمایش اولیه سریع‌تر:
python tvcl_real.py --data ".\dataset" --epochs 30 --patience 7 --seeds 0


۴. خروجی‌های مورد انتظار
- tvcl_real_ablation.csv: نتایج Full و Ablationها
- tvcl_real_summary.csv: میانگین و انحراف معیار
- tvcl_split_seed_0.csv: شناسه خودروها و تقسیم‌بندی داده
- فایل‌های Split برای Seedهای دیگر
۵. یک نکته درباره داده‌ای که فرستادید
فایل veh_3366.json را بررسی کردم. این فایل شامل ۱۹ پیام و ۳ شناسه فرستنده است. در حال حاضر فقط همین یک فایل در اختیار من قرار گرفته، نه کل ۸۴۱ فایل.

در طراحی جدید:
- فیلد attacker فقط برای برچسب استفاده می‌شود.
- sender_id برای تفکیک خودروها استفاده می‌شود.
- distance_to_road_edge و driversProfile وارد ویژگی‌های یادگیری نشده‌اند تا از استفاده از اطلاعات احتمالی شبیه‌ساز جلوگیری شود.
- هیچ داده مصنوعی تولید نمی‌شود.
- هیچ خودرو یا پیامی بین Train، Validation و Test مشترک نخواهد بود.
برای اجرای واقعی و بررسی عملکرد Full TVCL، لطفاً کل پوشه ۸۴۱ فایل را به‌صورت یک ZIP بارگذاری کنید. در آن صورت می‌توانم اجرای واقعی، خطاهای احتمالی استخراج ویژگی و نتایج هر هشت حالت Ablation را بررسی کنم.

فعلاً کد از نظر Syntax و خواندن فایل نمونه بررسی شده است؛ هنوز نتایج آموزشی روی مجموعه کامل ۸۴۱ فایل تولید نشده‌اند.