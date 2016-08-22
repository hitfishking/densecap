require 'torch'
require 'nn'
require 'image'
require 'hdf5'  -- ? 安装失败，尝试unbuntu server 64bit

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'  -- 变量尽可能是local的.
local box_utils = require 'densecap.box_utils'

-- 本文件只是lua的一个一般module，有公共函数和私有函数; 不是更高级的，类似DenseCapModel那样的类文件.
-- Torch项目作为lua项目，其就是由这两类文件(module)构成的.

local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint', 'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.4)
cmd:option('-num_proposals', 1000)  --? proposals数和的boxes数为什么不同?
cmd:option('-boxes_per_image', 100) --?

cmd:option('-input_txt', '')  -- 必要参数；存储待处理的image_path.
cmd:option('-max_images', 0)
cmd:option('-output_h5', '')  -- 必要参数; hdf5输出目标文件.

cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)


-- 什么地方调用的?
-- 准备image，输入model；得到boxes和features；
local function run_image(model, img_path, opt, dtype)  -- dtype指明image tensor的数据类型
  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)  -- 3是depth；[res] image.load(filename, [depth, tensortype])
  img = image.scale(img, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W) -- 使用view来3维图片的shape改成4维；此时的img_caffe物理上不存在.
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255) -- index函数抽取出相应项，给变量分配独立存储；255是什么?
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68} -- 这些值怎么得到的?对新image也适用吗?
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W) -- 复制H*W份；RGB每个值都要剪掉这些值.
  img_caffe:add(-1, vgg_mean) -- 图片数据均值归一；

  -- 真正的extractFeatures功能在model中；输入处理好的image，抽取出xcycwh坐标?；feats?
  local boxes_xcycwh, feats = model:extractFeatures(img_caffe:type(dtype))  -- 重心部分；image:type应该就是tensor的type，返回的是type名.
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh) -- 产生boxes的xywh坐标.
  return boxes_xywh, feats
end

-- 该模块应是在命令行中运行；main()比较规整，非必须(不同于C语言)；run_model.lua中就没使用main().
local function main()
  local opt = cmd:parse(arg)
  assert(opt.input_txt ~= '', 'Must provide -input_txt') -- Torch包提供了基本的测试能力；但没有测试框架.
  assert(opt.output_h5 ~= '', 'Must provide -output_h5')
  
  -- Read the text file of image paths
  local image_paths = {}
  for image_path in io.lines(opt.input_txt) do -- io包是lua系统更底层的包；lua语言自带；lua语言都有哪些系统包?
    table.insert(image_paths, image_path)      -- table也是lua系统级的包，可直接用.
    if opt.max_images > 0 and #image_paths == opt.max_images then -- 最多只从文件中读取opt.max_images个路径.
      break
    end
  end
  
  -- Load and set up the model
  local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn) -- utils是一个lua module.
  local checkpoint = torch.load(opt.checkpoint)  -- Torch File相关函数；High level file operations.
  local model = checkpoint.model  -- 文件存储了多个变量，其中之一是model；文件是怎么存储的? 还包含哪些字段?
  model:convert(dtype, use_cudnn) -- DenseCap model定义有14个函数(不算内部调用的大量其他模块)；
  model:setTestArgs{
    rpn_nms_thresh = opt.rpn_nms_thresh,
    final_nms_thresh = opt.final_nms_thresh,
    num_proposals = opt.num_proposals,
  }
  model:evaluate()   -- ? 什么功能? 输出什么?
  
  -- Set up the output tensors
  -- torch-hdf5 can only create datasets from tensors; there is no support for
  -- creating a dataset and then writing it in pieces. Therefore we have to
  -- keep everything in memory and then write it to disk ... gross.
  -- 13k images, 100 boxes per image, and 4096-dimensional features per box
  -- will take about 20GB of memory.
  local N = #image_paths
  local M = opt.boxes_per_image
  local D = 4096 -- TODO this is specific to VG  -- feature 的长度是4096
  local all_boxes = torch.FloatTensor(N, M, 4):zero()
  local all_feats = torch.FloatTensor(N, M, D):zero()
  
  -- Actually run the model; 循环执行run_image()函数，
  for i, image_path in ipairs(image_paths) do
    print(string.format('Processing image %d / %d', i, N))
    local boxes, feats = run_image(model, image_path, opt, dtype)  -- --这句是重心.
    all_boxes[i]:copy(boxes[{{1, M}}])  -- 受限于HDF5的输出要求，只能在内存中构建好整个输出数据tensor，再一次性写入HDF5文件.
    all_feats[i]:copy(feats[{{1, M}}])  --? 这种格式?
  end

  -- Write data to the HDF5 file
  -- hdf5文件只能使用tensor进行整体输出; 不能分块写入.
  -- 13k个image，100bbox/image，4096-d feature/bbox； 约需要20G内存.
  local h5_file = hdf5.open(opt.output_h5)
  h5_file:write('/feats', all_feats)
  h5_file:write('/boxes', all_boxes)
  h5_file:close()
end

main()
