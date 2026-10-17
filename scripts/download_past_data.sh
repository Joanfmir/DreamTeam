#!/usr/bin/env bash
# Baixa les dades públiques dels reptes anteriors de la Datathon FME a data/past/
# Font: https://github.com/data-students (Associació d'Estudiants de Dades)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/past"
RAW="https://raw.githubusercontent.com/data-students"
mkdir -p "$OUT"

fetch() {  # fetch <url> <destí>
  mkdir -p "$(dirname "$2")"
  if [ -f "$2" ]; then echo "  ja existeix: ${2#$ROOT/}"; return; fi
  echo "  baixant ${2#$ROOT/}"
  curl -fsSL "$1" -o "$2"
}

echo "▸ 2025 Schneider Electric (classificació + explicabilitat)"
fetch "$RAW/datathon2025-challenges/main/Schneider%20Electric/dataset.csv" "$OUT/2025_schneider/dataset.csv"
fetch "$RAW/datathon2025-challenges/main/Schneider%20Electric/statament.pdf" "$OUT/2025_schneider/enunciat.pdf"
fetch "$RAW/datathon2025-challenges/main/Mango/statement.pdf" "$OUT/2025_mango/enunciat.pdf"

echo "▸ 2022 Accenture (supply chain, AUC)"
for f in orders product_attributes cities_data test; do
  fetch "$RAW/datathon2022/main/accenture_challenge/datathon_SC_ACN_22/$f.csv" "$OUT/2022_accenture/$f.csv"
done

echo "▸ 2024 AED (formació d'equips)"
fetch "$RAW/AEDChallenge/main/data/datathon_participants.json" "$OUT/2024_aed/datathon_participants.json"

echo "▸ 2021 Reviews (NLP)"
fetch "$RAW/datathon2021/main/Reviews_Challenge/train_reviews.json" "$OUT/2021_reviews/train_reviews.json"
fetch "$RAW/datathon2021/main/Reviews_Challenge/test_reviews.json" "$OUT/2021_reviews/test_reviews.json"

echo "✓ Fet. Dades a data/past/"
echo "  Mango 2023 (outfits, ~imatges): https://mng-datathon-upc.s3.eu-west-1.amazonaws.com/datathon.zip (gran, baixeu-lo a mà)"
