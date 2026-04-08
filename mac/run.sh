#!/usr/bin/env bash
# Launch the Mac speaker detection app.
# 第一次行會自動整 venv、裝 dependencies、download ECAPA-TDNN model。
set -e

cd "$(dirname "$0")"
echo "==> 工作目錄：$(pwd)"

VENV_DIR=".venv"

# --- 檢查 python3 ----------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "錯誤：搵唔到 python3。"
  echo "請喺 Terminal 行以下 command 裝 Xcode Command Line Tools："
  echo "    xcode-select --install"
  echo "裝完之後再行一次 ./run.sh。"
  exit 1
fi

# --- 建立 venv -------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "==> 建立 Python 虛擬環境..."
  if ! python3 -m venv "$VENV_DIR"; then
    echo ""
    echo "錯誤：建立 venv 失敗。試下重新裝 Xcode Command Line Tools："
    echo "    xcode-select --install"
    exit 1
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- 安裝 dependencies -----------------------------------------------------
if [ ! -f ".deps_installed" ]; then
  echo "==> 安裝 dependencies（要等 3-5 分鐘，呢個係正常嘅）..."
  if ! pip install --upgrade pip; then
    echo ""
    echo "錯誤：升級 pip 失敗。請檢查網絡後再試一次。"
    exit 1
  fi
  if ! pip install -r requirements.txt; then
    echo ""
    echo "錯誤：安裝 dependencies 失敗。"
    echo "請檢查網絡（試下 ping google.com），然後再行一次 ./run.sh。"
    exit 1
  fi
  touch .deps_installed
  echo "==> Dependencies 裝完喇。"
fi

# --- 啟動 ------------------------------------------------------------------
echo "==> 啟動 Speaker Detect...（第一次要等 ECAPA-TDNN model 下載，大約 30 秒）"
exec python speaker_detect.py
