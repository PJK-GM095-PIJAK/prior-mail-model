"""Template-based synthetic Indonesian work-email generator (ML_PIPELINE.md §7).

A privacy-free way to bootstrap the internal labeled set: no real user data, no
PII, fully reproducible from a seed. Emits records in the §7 JSONL schema
(``src/data/labeled.py``) so they flow straight through ``make data``.

Design intent: the templates encode genuine *urgency* signals (deadlines,
account alerts, action-required vs. informational vs. bulk), NOT just topic.
This is deliberately the opposite weakness of the public English dataset, whose
topic->priority mapping is only a proxy. Synthetic data is still synthetic — it
does not replace real annotated email, but it lets the domain-adaptation path be
exercised end-to-end today.

CLI: ``python -m src.data.synthetic --n 200 --seed 42 --out data/labeled/synthetic_v1.jsonl``
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from src.data.labeled import validate_record
from src.utils.constants import PRIORITY_LABELS

logger = logging.getLogger(__name__)

# --- Slot vocabularies (Indonesian work context) --------------------------
NAMES = ["Budi", "Siti", "Andi", "Rina", "Dewi", "Agus", "Putri", "Joko", "Maya", "Fajar"]
PROJECTS = ["Proyek Mawar", "migrasi server", "kampanye Q3", "audit keuangan", "rilis aplikasi"]
TEAMS = ["tim Marketing", "tim Engineering", "divisi Keuangan", "tim Produk", "HRD"]
TIMES = ["pukul 09.00", "pukul 14.00 hari ini", "sebelum jam 5 sore", "besok pagi"]
DATES = ["Senin depan", "tanggal 15", "akhir minggu ini", "30 Mei"]
AMOUNTS = ["Rp1.500.000", "Rp250.000", "Rp10.000.000"]

# --- Templates per priority class -----------------------------------------
# Each entry: (subject_template, body_template). Urgency is in the CONTENT,
# not the topic, so the model learns priority rather than category.
TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "urgent": [
        ("URGENT: Server {project} down",
         "Halo {name}, server untuk {project} mati sejak {time}. Mohon segera ditangani, "
         "layanan pelanggan terdampak. Butuh respons sekarang."),
        ("Tindakan diperlukan: verifikasi akun",
         "Akun Anda terdeteksi login mencurigakan. Verifikasi {time} atau akun akan dikunci."),
        ("Deadline hari ini: {project}",
         "{name}, deliverable {project} jatuh tempo {time}. Tolong kirim sebelum batas waktu."),
    ],
    "high": [
        ("Mohon review proposal {project}",
         "Hai {name}, bisa tolong review proposal {project} dan beri masukan sebelum {date}? "
         "Saya tunggu balasannya. Terima kasih."),
        ("Konfirmasi kehadiran rapat {project}",
         "{name}, rapat {project} dijadwalkan {date}. Mohon konfirmasi kehadiran Anda."),
        ("Tagihan {amount} perlu persetujuan",
         "Halo {name}, ada tagihan {amount} dari {team} yang menunggu persetujuan Anda."),
    ],
    "normal": [
        ("Notulen rapat {team}",
         "Berikut notulen rapat {team} minggu ini. Tidak ada tindak lanjut khusus, hanya untuk arsip."),
        ("Update status {project}",
         "Sekadar info, {project} berjalan sesuai rencana. Update berikutnya {date}."),
        ("Struk pembayaran {amount}",
         "Pembayaran sebesar {amount} telah berhasil diproses. Ini adalah tanda terima Anda."),
    ],
    "low": [
        ("Promo spesial bulan ini!",
         "Dapatkan diskon hingga 50% untuk semua produk! Penawaran terbatas, jangan lewatkan."),
        ("Newsletter {team} edisi Mei",
         "Baca artikel terbaru dan tips dari {team} di buletin bulanan kami."),
        ("Anda punya rekomendasi baru",
         "Berdasarkan aktivitas Anda, kami pilihkan beberapa konten yang mungkin menarik."),
    ],
}


def _fill(text: str, rng: random.Random) -> str:
    return text.format(
        name=rng.choice(NAMES),
        project=rng.choice(PROJECTS),
        team=rng.choice(TEAMS),
        time=rng.choice(TIMES),
        date=rng.choice(DATES),
        amount=rng.choice(AMOUNTS),
    )


def generate(n: int, seed: int = 42) -> list[dict]:
    """Generate ``n`` synthetic labeled records, balanced across the 4 classes.

    Deterministic for a given ``(n, seed)``. Each record passes §7 validation.
    """
    rng = random.Random(seed)
    now = datetime.now().isoformat(timespec="seconds")
    records: list[dict] = []

    # Round-robin the classes so the set is balanced regardless of n.
    for i in range(n):
        label = PRIORITY_LABELS[i % len(PRIORITY_LABELS)]
        subject_t, body_t = rng.choice(TEMPLATES[label])
        record = {
            "id": f"syn-{seed}-{i:04d}",
            "subject": _fill(subject_t, rng),
            "body": _fill(body_t, rng),
            "label": label,
            "annotator": "synthetic",
            "labeled_at": now,
        }
        records.append(validate_record(record, source="synthetic", line=i))

    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %d synthetic records to %s", len(records), out_path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Indonesian labeled emails")
    parser.add_argument("--n", type=int, default=200, help="number of records (balanced across 4 classes)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    parser.add_argument(
        "--out", type=Path, default=Path("data/labeled/synthetic_v1.jsonl"), help="output JSONL path"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    write_jsonl(generate(args.n, seed=args.seed), args.out)


if __name__ == "__main__":
    _main()
