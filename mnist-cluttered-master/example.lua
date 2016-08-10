--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

local mnist_cluttered = require 'mnist_cluttered'

local dataConfig = {megapatch_w=100, num_dist=8}
local dataInfo = mnist_cluttered.createData(dataConfig)
require 'image'
fd = io.open('labels.txt', 'w')

for i = 1, 10 do
    local observation, target = unpack(dataInfo.nextExample())
    local formatted = image.toDisplayTensor({input=observation})
    image.save("imgs/" .. tostring(i) .. ".png", formatted)
    fd:write(tostring(target) .. '\n')
end

fd:close()