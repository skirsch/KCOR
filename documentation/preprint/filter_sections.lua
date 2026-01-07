-- Pandoc Lua filter to exclude supplement or main paper sections
-- This runs AFTER pandoc-crossref, so cross-references are already resolved
-- Set filter_mode via metadata file: filter_mode: "main" or filter_mode: "supplement"

local filter_mode = "main"
local in_supplement = false

function Meta(meta)
  -- Get filter mode from metadata (handles YAML string format)
  if meta.filter_mode then
    if type(meta.filter_mode) == "string" then
      filter_mode = meta.filter_mode
    elseif type(meta.filter_mode) == "table" and #meta.filter_mode > 0 then
      -- Handle array format
      if type(meta.filter_mode[1]) == "string" then
        filter_mode = meta.filter_mode[1]
      elseif type(meta.filter_mode[1]) == "table" and meta.filter_mode[1].text then
        filter_mode = meta.filter_mode[1].text
      end
    end
  end
end

function Header(el)
  -- Check if this is the supplement header (level 1 or 2)
  local text = pandoc.utils.stringify(el)
  local lower_text = text:lower()
  
  -- Match: "KCOR Supplementary material" or "Supplementary material"
  if lower_text:match("kcor.*supplementary") or lower_text:match("supplementary material") then
    in_supplement = true
    if filter_mode == "main" then
      return {}  -- Remove supplement header for main paper
    end
  end
  
  return el
end

function Block(el)
  -- Also check headers in Block function (some headers might not trigger Header function)
  if el.t == "Header" then
    local text = pandoc.utils.stringify(el)
    local lower_text = text:lower()
    if lower_text:match("kcor.*supplementary") or lower_text:match("supplementary material") then
      in_supplement = true
      if filter_mode == "main" then
        return {}  -- Remove supplement header
      end
    end
  end
  
  -- Filter blocks based on section
  if in_supplement then
    if filter_mode == "main" then
      return {}  -- Remove supplement blocks from main paper
    end
  else
    if filter_mode == "supplement" then
      return {}  -- Remove main paper blocks from supplement
    end
  end
  
  return el
end

function Inline(el)
  -- Preserve inline elements (cross-refs are inline)
  return el
end
