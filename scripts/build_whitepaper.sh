#!/bin/bash
set -e

echo "📄 Building NeuroShard Whitepaper"
echo "=================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHITEPAPER_DIR="$SCRIPT_DIR/docs/whitepaper"
PROTECTED_DIR="$SCRIPT_DIR/website/protected"

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex is not installed."
    echo "   Please install a LaTeX distribution (e.g., texlive-full on Ubuntu)"
    exit 1
fi

# Change to whitepaper directory
cd "$WHITEPAPER_DIR"

echo "[1/3] Compiling LaTeX (first pass)..."
pdflatex -interaction=nonstopmode neuroshard_whitepaper.tex > /dev/null 2>&1 || {
    echo "❌ LaTeX compilation failed. Check neuroshard_whitepaper.log for details."
    exit 1
}

echo "[2/3] Compiling LaTeX (second pass for references)..."
pdflatex -interaction=nonstopmode neuroshard_whitepaper.tex > /dev/null 2>&1 || {
    echo "❌ LaTeX compilation failed. Check neuroshard_whitepaper.log for details."
    exit 1
}

# Check if PDF was generated
if [ ! -f "neuroshard_whitepaper.pdf" ]; then
    echo "❌ Error: PDF was not generated."
    exit 1
fi

echo "[3/3] Copying PDF to protected website folder..."
mkdir -p "$PROTECTED_DIR"
cp neuroshard_whitepaper.pdf "$PROTECTED_DIR/whitepaper.pdf"

# Get file size for confirmation
PDF_SIZE=$(du -h "$PROTECTED_DIR/whitepaper.pdf" | cut -f1)

echo "=================================="
echo "✅ Whitepaper built successfully!"
echo "   Location: $PROTECTED_DIR/whitepaper.pdf"
echo "   Size: $PDF_SIZE"
echo ""
echo "📌 Note: The whitepaper is now protected and only"
echo "   accessible to registered users via /api/whitepaper/pdf"

