require 'torch'
require 'util.SequenceFactory'




local next_batch_eoi = torch.TestSuite()
local tester = torch.Tester()

local redis = require 'redis'
local red = redis.connect('127.0.0.1', 6379)

local ExternalMinibatchLoader_NextEventOfInterest = require 'util.ExternalMinibatchLoader_NextEventOfInterest'

opt = {}
opt.redis_prefix = "test"
opt.gpuid = -1
opt.horizon = 60
opt.num_time_slots = 60
opt.sequence_provider = 1
opt.redis_prefix = "test"

factory_initialization()

red:sadd("test-events-of-interest", "123") 

local loader = ExternalMinibatchLoader_NextEventOfInterest.create() 


function next_batch_eoi.no_eoi()
  local seq = "123-3,234-4,345-5"
  red:lpush("test-train", seq)
  local x, y, e_x, e_y = loader:next_batch() 
  local y_fasit = torch.IntTensor(1, 2, 1):fill(opt.num_time_slots)
  tester:eq(y, y_fasit,  "y:" .. tostring(y) .. "y_fasit:" .. tostring(y_fasit))
  local e_y_fasit = torch.IntTensor(1, 2, 1):fill(1)
  tester:eq(e_y, e_y_fasit, "e_y:" .. tostring(e_y) .. "e_y_fasit:" .. tostring(e_y_fasit))
end

function next_batch_eoi.one_eoi_in_1st_pos()
  local seq = "123-123,234-4,345-5"
  red:lpush("test-train", seq)
  local x, y, e_x, e_y = loader:next_batch() 
  local y_fasit = torch.IntTensor(1, 2, 1):fill(opt.num_time_slots)
  tester:eq(y, y_fasit,  "y:" .. tostring(y) .. "y_fasit:" .. tostring(y_fasit)) 
  local e_y_fasit = torch.IntTensor(1, 2, 1):fill(1)
  tester:eq(e_y, e_y_fasit, "e_y:" .. tostring(e_y) .. "e_y_fasit:" .. tostring(e_y_fasit))
end

function next_batch_eoi.one_eoi_in_2nd_pos()
  local seq = "120-1,130-123,140-5,150-5"
  red:lpush("test-train", seq)
  local x, y, e_x, e_y = loader:next_batch()
  local y_fasit = torch.IntTensor({{{11},{1},{opt.num_time_slots}}})
  tester:eq(y, y_fasit,  "y:" .. tostring(y) .. "y_fasit:" .. tostring(y_fasit))
  local e_y_fasit = torch.IntTensor({{{2},{1},{1}}})
  tester:eq(e_y, e_y_fasit, "e_y:" .. tostring(e_y) .. "e_y_fasit:" .. tostring(e_y_fasit))
end


function next_batch_eoi.one_eoi_in_3rd_pos()
  local seq = "120-1,130-13,140-123,150-5,160-5"
  red:lpush("test-train", seq)
  local x, y, e_x, e_y = loader:next_batch()
  local y_fasit = torch.IntTensor({{{21},{11},{1},{opt.num_time_slots}}})
  tester:eq(y, y_fasit, "y:" .. tostring(y) .. "y_fasit:" .. tostring(y_fasit))
  local e_y_fasit = torch.IntTensor({{{2},{2},{1},{1}}})
  tester:eq(e_y, e_y_fasit, "e_y:" .. tostring(e_y) .. "e_y_fasit:" .. tostring(e_y_fasit))
end


function next_batch_eoi.two_eois_in_middle()
  local seq = "120-1,130-13,140-123,150-5,160-123,170-5,180-5"
  red:lpush("test-train", seq)
  local x, y, e_x, e_y = loader:next_batch()
  local y_fasit = torch.IntTensor({{{21},{11},{1},{11},{1},{opt.num_time_slots}}})
  tester:eq(y, y_fasit, "y:" .. tostring(y) .. "y_fasit:" .. tostring(y_fasit))
  local e_y_fasit = torch.IntTensor({{{2},{2},{2},{2},{1},{1}}})
  tester:eq(e_y, e_y_fasit, "e_y:" .. tostring(e_y) .. "e_y_fasit:" .. tostring(e_y_fasit))
end

function next_batch_eoi.two_eois_with_gap_in_middle()
  local seq = "120-1,130-13,140-123,250-5,260-123,270-5,280-5"
  red:lpush("test-train", seq)
  local x, y, e_x, e_y = loader:next_batch()
  local y_fasit = torch.IntTensor({{{21},{11},{1},{11},{1},{opt.num_time_slots}}})
  tester:eq(y, y_fasit, "y:" .. tostring(y) .. "y_fasit:" .. tostring(y_fasit))
  local e_y_fasit = torch.IntTensor({{{2},{2},{2},{2},{1},{1}}})
  tester:eq(e_y, e_y_fasit, "e_y:" .. tostring(e_y) .. "e_y_fasit:" .. tostring(e_y_fasit))
end

local misc = torch.TestSuite()

function misc.PCA()
  -- Random 2D data with std ~(1.5,6)
  local N = 100
  math.randomseed(os.time())
  local x1 = torch.randn(N) * 1.5 + math.random()
  local x2 = torch.randn(N) * 6 + 2 * math.random()
  local X = torch.cat(x1, x2, 2) -- Nx2

  -- Rotating the data randomly
  local theta = math.random(180) * math.pi / 180
  local R = torch.Tensor{
    {math.cos(theta), -math.sin(theta)},
    {math.sin(theta),  math.cos(theta)}
  }
  X = X * R:t()
  X[{ {},1 }]:add(25)
  X[{ {},2 }]:add(10)
  
  vv = PCA(X)
  
  local deltas = vv[{ {1,1} , {3,4} }][1]
  local a = deltas[2]/deltas[1]
  local M_point = vv[{ {1,1} , {1,2} }][1]
  local b = M_point[2] - a*M_point[1]
  
  local p_x = torch.linspace(10, 40)
  local p_y = p_x:clone():mul(a):add(b)
  
  gnuplot.plot{
   {'dataset',X,'+'},
   {'PC1',vv[{ {1,1} , {} }],'v'},
   {'PC2',vv[{ {2,2} , {} }],'v'},
   {"myPCA", p_x, p_y,'-'}
  }
  gnuplot.axis('equal')
  gnuplot.axis{-20,50,-10,30}
  gnuplot.grid(true)
  --sys.sleep(1000)
end



tester:add(next_batch_eoi)
--tester:add(misc)

tester:run()

-- tier down
red:del("test-events-of-interest")


