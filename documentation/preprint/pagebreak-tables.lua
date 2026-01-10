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

function Pandoc(doc)
  local pb = pagebreak(FORMAT)
  if pb.t == "Null" then
    return doc
  end

  local out = {}
  local prev = nil

  for _, block in ipairs(doc.blocks) do
    if block.t == "Table" then
      -- If the immediately preceding emitted block is a "Table:" caption paragraph,
      -- treat it as part of the table so caption + table always stay together.
      local caption = nil
      if #out > 0 and is_table_caption_para(out[#out]) then
        caption = table.remove(out)
        prev = out[#out]
      end

      local prev_is_header = prev and prev.t == "Header"
      local prev_is_pb = is_pagebreak(prev)

      if prev and (not prev_is_header) and (not prev_is_pb) then
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

