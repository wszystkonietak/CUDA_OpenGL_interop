require("cuda-exported-variables")

if os.target() == "windows" then
    dofile("premake_scripts/premake5-cuda-vs.lua")
elseif os.target() == "linux" then
    dofile("premake_scripts/premake5-cuda-nvcc.lua")
end
