# Paper Build System

This directory uses a **single source file** (`paper.md`) to generate a **single** Word document (`paper.docx`) and a **single** PDF (`paper.pdf`).

## Architecture

### Single Source File
- **`paper.md`**: The full manuscript (including appendices), built as a single continuous document (no split/separators).

### Formatting Filters
- **`pagebreak-tables.lua`**: Inserts page breaks before and after each table so every table appears on its own page in DOCX and PDF builds.

### Build Targets

#### Word Documents (DOCX)
- **`make paper-docx`**: Builds `paper.docx`
- Uses `pandoc-crossref-docx.yaml`

#### PDF Documents
- **`make paper-pdf`**: Builds `paper.pdf`

#### Combined Build
- **`make paper`**: Builds `paper.docx` and `paper.pdf`, then copies to website

## Example Commands

```bash
# Build PDF
make paper-pdf

# Build Word document
make paper-docx

# Build both + copy to website
make paper
```

## Migration Notes

Legacy split-document tooling (`drop-div.lua`, `pandoc-crossref-docx-main.yaml`, `pandoc-crossref-docx-supplement.yaml`, `split_docx.py`) is kept for reference but is no longer used by the root `Makefile` paper targets.

