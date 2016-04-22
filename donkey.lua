require 'lfs'
local BUFSIZE = 2^13     -- 8K
local eois = {}

function loadSourceNames(path)
  for fname in lfs.dir(path) do  
    if lfs.attributes(fname,"mode") == nil then
    table.insert(eois, fname)
    end
  end
end


local function loadFile(path)
  local f = io.input(path)
  
  local data, rest = f:read(BUFSIZE, "*line")
  if rest then data = data .. rest .. '\n' end
  
  lines={}
  for line in data:gmatch("[^\r\n]+") do 
    table.insert(lines, line)
  end
  
  
  return lines
end

function loadEvents(path, redis_client)
  local lines = loadFile(path)
  for i=1, #lines do
    redis_client:sadd(opt.redis_prefix .. "-events", lines[i])
  end
end


local function splitLines(lineTable, keep_k)
  local pos = 1
  
  while #lineTable > keep_k do
    pos = torch.random(1, #lineTable)
    table.remove(lineTable, pos)
  end
  
  for i=1, #lineTable do
    lineTable[i] = string.split(lineTable[i], ",")
    lineTable[i][2] = opt.event_mapping[lineTable[i][2]]
  end
  
  return lineTable
end


local function splitLines2(lineTable, keep_k)
  local pos = 1
  
  if #lineTable > keep_k then
    pos = torch.random(1, #lineTable - keep_k)
  end
  
  local tmp ={}
  
  for i=1, math.min(keep_k, #lineTable) do
    table.insert(tmp, string.split(lineTable[pos+i-1], ","))
    
    tmp[i][2] = opt.event_mapping[tmp[i][2]]
  end
  return tmp
end

local function insertProbes(seq, percentage)
  local len_seq = math.ceil(#seq*percentage)

  while #seq < len_seq do
    local pos = torch.random(1, #seq-1)
    local t = torch.random(seq[pos][1], seq[pos+1][1])
    table.insert(seq, pos+1, {tostring(t), 1})
  end
  
end 

local function formatBatch(batch)
  for i=1, #batch do
    for j= 1, #batch[i] do
      batch[i][j] = batch[i][j][1] .. "-" .. batch[i][j][2]
    end
  end
end

function loadBatch(size, len_seq)
  local batch = {}
  len_seq = len_seq + 1  
  while #batch < size do  
    local eoi = torch.random(1, #eois)
    local files = loadFile("2013.csv_eoi/" .. eois[eoi])
    local file = torch.random(1, #files)
    --print ( "2013.csv_sources/" .. files[file])
    local seq = loadFile("2013.csv_sources/" .. files[file])
    
    if #seq > math.ceil(len_seq*1.2) then
      table.remove(seq, 1) -- remove this line when Axel has fixed the "timestamp, event" from all files
      --if torch.random(1, 2) == 1 then
      --  seq = splitLines(seq, math.ceil(len_seq*1.2))
      --else
        seq = splitLines2(seq, len_seq)
        --insertProbes(seq, 1.2)
      --end
      --print ("seq", seq)
      table.insert(batch, seq)
    else
      table.remove(files, file)
    end
  end
  --insertProbes(batch)
  --formatBatch(batch)
  --print ("batch", batch)
  collectgarbage()
  return batch
end

