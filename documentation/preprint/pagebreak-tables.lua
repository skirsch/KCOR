-- pagebreak-tables.lua
-- Put each table on its own page, but only insert a page break BEFORE a table
-- when needed (i.e., avoid pushing the first table away from a "Tables" header).
--
-- Rules:
-- - Do NOT insert a page break AFTER tables (so any notes/text immediately
--   following a table can remain on the same page).
-- - Insert a page break BEFORE a table only if the previous meaningful block is:
--     - not a page break, AND
--     - not a Header (so "Tables" heading can stay with the first table).
--
-- DOCX: emits OOXML page breaks
-- PDF (LaTeX): emits \newpage

local function pagebreak(format)
  if format == "docx" then
    return pandoc.RawBlock("openxml", '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
  elseif format == "latex" then
    return pandoc.RawBlock("tex", "\\newpage")
  else
    -- No-op for other formats (html, etc.)
    return pandoc.Null()
  end
end

local function is_pagebreak(block)
  if not block then return false end
  if block.t ~= "RawBlock" then return false end
  if block.format == "openxml" and type(block.text) == "string" then
    return block.text:match('w:type="page"') ~= nil
  end
  if (block.format == "tex" or block.format == "latex") and type(block.text) == "string" then
    -- common pagebreak commands
    return block.text:match("\\newpage") or block.text:match("\\pagebreak") or block.text:match("\\clearpage")
  end
  return false
end

local function is_table_caption_para(block)
  if not block or block.t ~= "Para" then return false end
  local s = pandoc.utils.stringify(block)
  -- Pandoc table captions in this repo use "Table: ..." immediately before the table.
  return s:match("^Table:%s+") ~= nil
end

local function is_blank_para(block)
  if not block or block.t ~= "Para" then return false end
  local s = pandoc.utils.stringify(block)
  return s:match("^%s*$") ~= nil
end

local function is_table_numbering_preamble(block)
  -- We sometimes insert a raw LaTeX snippet immediately under an "Appendix X tables"
  -- header to reset table numbering (e.g., \renewcommand{\thetable}{D.\arabic{table}}
  -- and \setcounter{table}{0}). Treat that snippet as part of the header group so
  -- the first table doesn't get pushed to a new page.
  if not block or block.t ~= "RawBlock" then return false end
  if not (block.format == "tex" or block.format == "latex") then return false end
  if type(block.text) ~= "string" then return false end
  return block.text:match("\\renewcommand%s*%{%s*\\thetable%s*%}") ~= nil
      or block.text:match("\\setcounter%s*%{%s*table%s*%}%s*%{") ~= nil
end

local function is_appendix_tables_header(block)
  if not block or block.t ~= "Header" then return false end
  -- We want a page break BEFORE headings like:
  --   ### Appendix C tables
  -- so that each appendix table block starts on its own page, while still
  -- allowing the first table to remain on the same page as the header.
  local s = pandoc.utils.stringify(block)
  return s:match("^Appendix%s+[A-Z]%s+tables%s*$") ~= nil
end

function Pandoc(doc)
  local pb = pagebreak(FORMAT)
  if pb.t == "Null" then
    return doc
  end

  local out = {}
  local prev = nil

  local function last_meaningful_idx()
    for i = #out, 1, -1 do
      local b = out[i]
      if b and b.t ~= "Null" and (not is_blank_para(b)) then
        return i
      end
    end
    return nil
  end

  for _, block in ipairs(doc.blocks) do
    if is_appendix_tables_header(block) then
      local prev_is_pb = is_pagebreak(prev)
      if prev and (not prev_is_pb) then
        table.insert(out, pb)
      end
      table.insert(out, block)
      prev = block
    elseif block.t == "Table" then
      -- If the immediately preceding emitted block is a "Table:" caption paragraph,
      -- treat it as part of the table so caption + table always stay together.
      local caption = nil
      if #out > 0 and is_table_caption_para(out[#out]) then
        caption = table.remove(out)
        prev = out[#out]
      end

      -- Determine the meaningful block immediately before the table (skipping blank
      -- lines and accounting for optional table-numbering preamble blocks).
      local idx = last_meaningful_idx()
      local prev_meaningful = idx and out[idx] or nil
      local prev_is_pb = is_pagebreak(prev_meaningful)

      local prev_is_header_like = prev_meaningful and prev_meaningful.t == "Header"
      if (not prev_is_header_like) and is_table_numbering_preamble(prev_meaningful) then
        local idx2 = (idx and idx > 1) and (idx - 1) or nil
        local prev2 = idx2 and out[idx2] or nil
        -- If the preamble sits right under a header, treat the whole thing as header-like.
        prev_is_header_like = prev2 and prev2.t == "Header"
      end

      if prev_meaningful and (not prev_is_header_like) and (not prev_is_pb) then
        table.insert(out, pb)
      end

      if caption then
        table.insert(out, caption)
      end
      table.insert(out, block)
      prev = block
    else
      table.insert(out, block)
      -- track the last non-Null block we emitted
      prev = block
    end
  end

  return pandoc.Pandoc(out, doc.meta)
end

