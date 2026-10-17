#!/usr/bin/env bash
# Baixa les dades públiques dels reptes anteriors de la Datathon FME a data/past/
#
#   bash scripts/download_past_data.sh          # tot (~125 MB)
#   bash scripts/download_past_data.sh schneider accenture   # només alguns reptes
#
# Fonts: https://github.com/data-students (Associació d'Estudiants de Dades) i, per a
# Mango 2024, el repo públic del 2n classificat (bielnebot/datathon_fme_2024).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/past"
GH="https://raw.githubusercontent.com"
AED="$GH/data-students"
mkdir -p "$OUT"

fetch() {  # fetch <url> <destí>
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then echo "  ✓ ja existeix: ${2#"$ROOT"/}"; return; fi
  echo "  ↓ ${2#"$ROOT"/}"
  curl -fsSL --retry 3 "$1" -o "$2" || { echo "  ✗ error baixant $1"; rm -f "$2"; }
}

want() {  # sense arguments → tot; amb arguments → només els reptes indicats
  [ $# -eq 0 ] && return 0
  for a in "${SELECTED[@]}"; do [[ "$1" == *"$a"* ]] && return 0; done
  return 1
}
SELECTED=("$@")
SEL() { if [ ${#SELECTED[@]} -eq 0 ]; then return 0; fi; want "$1"; }

if SEL schneider; then
  echo "▸ 2025 Schneider Electric · classificació + explicabilitat"
  fetch "$AED/datathon2025-challenges/main/Schneider%20Electric/dataset.csv" "$OUT/2025_schneider/dataset.csv"
  fetch "$AED/datathon2025-challenges/main/Schneider%20Electric/statament.pdf" "$OUT/2025_schneider/enunciat.pdf"
fi

if SEL mango2025; then
  echo "▸ 2025 Mango · previsió de demanda (només l'enunciat; les dades eren privades)"
  fetch "$AED/datathon2025-challenges/main/Mango/statement.pdf" "$OUT/2025_mango/enunciat.pdf"
fi

if SEL mango2024; then
  echo "▸ 2024 Mango · atributs de roba (metadades, sense imatges)"
  for f in product_data attribute_data test_data; do
    fetch "$GH/bielnebot/datathon_fme_2024/main/data/$f.csv" "$OUT/2024_mango/$f.csv"
  done
fi

if SEL aed; then
  echo "▸ 2024 AED · formació d'equips"
  fetch "$AED/AEDChallenge/main/data/datathon_participants.json" "$OUT/2024_aed/datathon_participants.json"
  fetch "$AED/AEDChallenge/main/participant.py" "$OUT/2024_aed/participant.py"
fi

if SEL ntt; then
  echo "▸ 2023 NTT Data · compres hospitalàries"
  fetch "$AED/datathon2023/main/ntt-data/consumo_material_clean.xlsx" "$OUT/2023_ntt/consumo_material_clean.xlsx"
  fetch "$AED/datathon2023/main/ntt-data/challenge.pptx" "$OUT/2023_ntt/enunciat.pptx"
fi

if SEL upc; then
  echo "▸ 2023 UPC Sostenible · mobilitat"
  fetch "$AED/datathon2023/main/upc-sostenible/challenge.pdf" "$OUT/2023_upc/enunciat.pdf"
  fetch "$AED/datathon2023/main/upc-sostenible/Datathon_Results_MOBILITY_2022_original_Students.xlsx" "$OUT/2023_upc/enquesta_mobilitat_2022.xlsx"
  fetch "$AED/datathon2023/main/upc-sostenible/Datathon_Students_enrolled_in_bachelors_and_masters_degrees_22-23_place_of_residence_course_V2.xlsx" "$OUT/2023_upc/estudiants_residencia_22-23.xlsx"
fi

if SEL accenture; then
  echo "▸ 2022 Accenture · supply chain (AUC)"
  for f in orders product_attributes cities_data test; do
    fetch "$AED/datathon2022/main/accenture_challenge/datathon_SC_ACN_22/$f.csv" "$OUT/2022_accenture/$f.csv"
  done
fi

if SEL qualcomm; then
  echo "▸ 2022 Qualcomm · optimització de cadenes de power switches"
  Q="$AED/datathon2022/main/qualcomm_challenge"
  fetch "$Q/examples/example_input.def" "$OUT/2022_qualcomm/example_input.def"
  fetch "$Q/examples/example_output.def" "$OUT/2022_qualcomm/example_output.def"
  fetch "$Q/solution_metrics.py" "$OUT/2022_qualcomm/solution_metrics.py"
  for i in 0 1 2 3; do fetch "$Q/new_testcases/priv_testcase$i.def" "$OUT/2022_qualcomm/priv_testcase$i.def"; done
fi

if SEL reviews; then
  echo "▸ 2021 Reviews · NLP sobre ressenyes de restaurants"
  fetch "$AED/datathon2021/main/Reviews_Challenge/train_reviews.json" "$OUT/2021_reviews/train_reviews.json"
  fetch "$AED/datathon2021/main/Reviews_Challenge/test_reviews.json" "$OUT/2021_reviews/test_reviews.json"
fi

echo "✓ Fet. Dades a data/past/ ($(du -sh "$OUT" | cut -f1))"
