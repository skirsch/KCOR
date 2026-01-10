# Paper Build System

This directory uses a **single source file** (`paper.md`) to generate a **single** LaTeX file (`paper.tex`) and a **single** PDF (`paper.pdf`).

## Architecture

### Single Source File
- **`paper.md`**: The full manuscript (including appendices), built as a single continuous document (no split/separators).

### Formatting Filters
- **`pagebreak-tables.lua`**: Inserts page breaks before and after each table so every table appears on its own page in DOCX and PDF builds.

### Build Targets

#### LaTeX
- **`make paper-tex`**: Builds `paper.tex`

#### PDF
- **`make paper-pdf`**: Builds `paper.pdf`

#### Combined Build
- **`make paper`**: Builds `paper.tex` and `paper.pdf`, then copies **only the PDF** to the website

## Example Commands

```bash
# Build PDF
make paper-pdf

# Build LaTeX
make paper-tex

# Build both + copy PDF to website
make paper
```

## Migration Notes

Legacy DOCX tooling (`pandoc-crossref-docx.yaml`, `drop-div.lua`, `pandoc-crossref-docx-main.yaml`, `pandoc-crossref-docx-supplement.yaml`, `split_docx.py`) is kept for reference but is no longer used by the root `Makefile` paper targets.

