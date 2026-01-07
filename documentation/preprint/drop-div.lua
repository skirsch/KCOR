-- drop-div.lua
-- Usage: pandoc -L drop-div.lua -M drop_div=supp ...
-- Drops Divs with identifier equal to metadata 'drop_div' (comma-separated list allowed)
--
-- This filter runs AFTER pandoc-crossref, so cross-references are already resolved.
-- The divs are dropped during AST transformation, preserving cross-reference numbering.

local function split_csv(s)
  local t = {}
  if not s then return t end
  -- Handle both string and table inputs
  local str = s
  if type(s) == "table" then
    if s.text then
      str = s.text
    elseif #s > 0 then
      str = s[1]
    else
      return t
    end
  end
  -- Convert to string if needed
  if type(str) ~= "string" then
    str = tostring(str)
  end
  -- Split by comma
  for item in string.gmatch(str, "([^,%s]+)") do
    table.insert(t, item)
  end
  return t
end

-- Store which divs to drop
local drop = {}

function Meta(meta)
  local v = meta.drop_div
  if v then
    local ids = split_csv(v)
    for _, id in ipairs(ids) do
      drop[id] = true
    end
  end
  return meta
end

function Pandoc(doc)
  -- Ensure we have the metadata (in case Meta wasn't called first)
  local meta = doc.meta
  if meta and meta.drop_div then
    local v = meta.drop_div
    local ids = split_csv(v)
    for _, id in ipairs(ids) do
      drop[id] = true
    end
  end
  
  -- Use pandoc.walk_block to recursively remove divs throughout the document
  local function remove_divs(blocks)
    local result = {}
    for i, el in ipairs(blocks) do
      if el.t == "Div" and el.identifier and drop[el.identifier] then
        -- Skip this div (don't add it to result)
      else
        -- Recursively process nested blocks
        if el.content and type(el.content) == "table" then
          el.content = remove_divs(el.content)
        end
        if el.blocks then
          el.blocks = remove_divs(el.blocks)
        end
        table.insert(result, el)
      end
    end
    return result
  end
  
  return pandoc.Pandoc(remove_divs(doc.blocks), doc.meta)
end
