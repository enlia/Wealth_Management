#!/bin/bash
# 批量下载腾讯财经行情（含名称、PE、PB、市值、换手率）
set -u
cd "$(dirname "$0")"
mkdir -p raw

LIST="${1:-codes_all.txt}"
PREFIX="${2:-all}"
BATCH=60
PAR=10

total=$(wc -l < "$LIST")
nb=$(( (total + BATCH - 1) / BATCH ))
echo "=== $LIST : $total 个标的, $nb 批, 并发 $PAR ==="

OUT="raw/${PREFIX}_raw.txt"
: > "$OUT"
rm -rf "raw/batch_${PREFIX}"; mkdir -p "raw/batch_${PREFIX}"
split -l $BATCH -d -a 4 "$LIST" "raw/batch_${PREFIX}_"

for f in raw/batch_${PREFIX}_*; do
  # 注意：Git Bash 下 split 产物含 CRLF，必须先剥掉 \r，否则 URL 非法、curl 返回空
  codes=$(tr -d '\r' < "$f" | grep -v '^[[:space:]]*$' | tr '\n' ',' | sed 's/,$//')
  [ -z "$codes" ] && continue
  curl -s --max-time 30 "http://qt.gtimg.cn/q=${codes}" >> "$OUT"
  printf '\n' >> "$OUT"
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do wait -n 2>/dev/null || break; done
done
wait

lines=$(wc -l < "$OUT")
hits=$(grep -o 'v_[a-z][0-9]*=' "$OUT" | wc -l)
echo "完成: 原始行 $lines, 命中记录 $hits / 期望 $total"
echo "文件: $OUT ($(du -h "$OUT" | cut -f1))"
