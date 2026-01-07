# Paper Build System

This directory uses a **single source file** (`paper.md`) with conditional section hiding to generate separate outputs while preserving cross-reference numbering.

## Architecture

### Single Source File
- **`paper.md`**: Contains both main text and supplementary material
- Main text is wrapped in `::: {#main}` div
- Supplementary material is wrapped in `::: {#supp}` div

### Filter System
- **`drop-div.lua`**: Lua filter that conditionally drops divs by ID based on metadata
- Runs **after** `pandoc-crossref`, so all cross-references are resolved before sections are hidden
- Uses `drop_div` metadata variable (comma-separated list of div IDs to drop)

### Build Targets

#### Word Documents (DOCX)
- **`make paper-docx`**: Builds both `paper.docx` (main) and `paper_supplement.docx`
- Uses metadata files:
  - `pandoc-crossref-docx-main.yaml` (sets `drop_div: "supp"`)
  - `pandoc-crossref-docx-supplement.yaml` (sets `drop_div: "main"`)

#### PDF Documents
- **`make paper-pdf`**: Builds `paper.pdf` (combined: main + supplement)
- **`make paper-pdf-main`**: Builds `paper_main.pdf` (main only, drops supplement)
- **`make paper-pdf-supplement`**: Builds `paper_supplement.pdf` (supplement only, drops main)

#### Combined Build
- **`make paper`**: Builds all DOCX and combined PDF, then copies to website

## How It Works

1. **Single build context**: All content exists when `pandoc-crossref` runs, so all labels/counters resolve correctly
2. **Conditional hiding**: The `drop-div.lua` filter removes divs by ID **after** cross-references are resolved
3. **Separate outputs**: Different builds drop different divs, producing separate PDFs/DOCX files

## Example Commands

```bash
# Build combined PDF (for preprints/sharing)
make paper-pdf

# Build main-only PDF (for journal submission)
make paper-pdf-main

# Build supplement-only PDF (for journal submission)
make paper-pdf-supplement

# Build all Word documents
make paper-docx

# Build everything
make paper
```

## Cross-Reference Preservation

Because the filter runs **after** `pandoc-crossref`, all cross-references are resolved in the same numbering context:
- Figure numbers are consistent across builds
- Equation numbers are sequential
- Table numbers match across outputs
- References to figures/equations/tables in the main text resolve correctly even when the supplement is dropped

## Migration Notes

The old `filter_sections.lua` (header-based detection) has been replaced with `drop-div.lua` (div-based). The old filter is kept for reference but is no longer used in the build system.

