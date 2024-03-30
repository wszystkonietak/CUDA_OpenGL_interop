require("/premake_scripts/premake5-cuda")

workspace "CUDA_OpenGL_interop"
	architecture "x86_64"
	startproject "CUDA_OpenGL_interop"
	configurations { "Debug", "Release" }
	warnings "Extra"
	flags
	{
    	"RelativeLinks",
		"MultiProcessorCompile"
	}
	filter "configurations:release"
    	symbols "Off"
	    optimize "Full"
    	runtime "Release"

  	filter "configurations:debug"
    	defines {"DEBUG"}
    	symbols "On"
    	optimize "Off"
    	runtime "Debug"
  	filter {}

outputbindir = "bin/%{cfg.system}-%{cfg.architecture}-%{cfg.buildcfg}/%{prj.name}"
outputobjdir = "bin-int/%{cfg.system}-%{cfg.architecture}-%{cfg.buildcfg}/%{prj.name}"
main_prj_name = "CUDA_OpenGL_interop"
project (main_prj_name)
	location "%{prj.name}"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"
	systemversion "latest"
	targetdir (outputbindir)
	objdir (outputobjdir)
	buildcustomizations "BuildCustomizations/CUDA 12.1"
	externalwarnings "Off"
	if os.target() == "windows" then
		cudaFiles { main_prj_name .. "/src/**.cu" } -- files to be compiled into binaries by VS CUDA.
	else
		toolset "nvcc"
		cudaPath "/usr/local/cuda"
		files { main_prj_name .. "/src/**.cu" }
		rules {"cu"}
	end
	cudaFastMath "On"
  	cudaRelocatableCode "On"
  	cudaVerbosePTXAS "On"
  	cudaMaxRegCount "32"
	cudaIntDir "../bin-int/%{cfg.system}-%{cfg.architecture}-%{cfg.buildcfg}/cudaobj"

	files
	{
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/src/**.txt",
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.hpp",
		"%{prj.name}/src/**.cuh",
	}

	includedirs
	{
		"%{wks.location}/vendor/glad/include",
		"%{wks.location}/vendor/glfw/include",
		"%{wks.location}/vendor/glm",
	}

	defines
	{
		"GLFW_INCLUDE_NONE"
	}

	links
	{
		"glad",
		"glfw"
	}


	filter "system:macosx"
		links
		{
			"CoreFoundation.framework",
			"Cocoa.framework",
			"IOKit.framework",
			"CoreVideo.framework"
		}

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"

group "Dependencies"

project "glad"
	location "vendor/glad"
	kind "StaticLib"
	language "C"
	staticruntime "on"
	systemversion "latest"

	targetdir (outputbindir)
	objdir (outputobjdir)

	files
	{
		"%{prj.location}/src/glad.c"
	}

	includedirs
	{
		"%{prj.location}/include"
	}

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"

project "glfw"
	location "vendor/glfw"
	kind "StaticLib"
	language "C"
	staticruntime "on"
	systemversion "latest"

	targetdir (outputbindir)
	objdir (outputobjdir)

	files
	{
		"%{prj.location}/include/GLFW/glfw3.h",
    "%{prj.location}/include/GLFW/glfw3native.h",
    "%{prj.location}/src/internal.h",
    "%{prj.location}/src/platform.h",
    "%{prj.location}/src/mappings.h",
    "%{prj.location}/src/context.c",
    "%{prj.location}/src/init.c",
    "%{prj.location}/src/input.c",
    "%{prj.location}/src/monitor.c",
    "%{prj.location}/src/platform.c",
    "%{prj.location}/src/vulkan.c",
    "%{prj.location}/src/window.c",
    "%{prj.location}/src/egl_context.c",
    "%{prj.location}/src/osmesa_context.c",
    "%{prj.location}/src/null_platform.h",
    "%{prj.location}/src/null_joystick.h",
    "%{prj.location}/src/null_init.c",

    "%{prj.location}/src/null_monitor.c",
    "%{prj.location}/src/null_window.c",
    "%{prj.location}/src/null_joystick.c",
	}

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"

	filter "system:windows"
		files
		{
			"%{prj.location}/src/win32_init.c",
        "%{prj.location}/src/win32_module.c",
        "%{prj.location}/src/win32_joystick.c",
        "%{prj.location}/src/win32_monitor.c",
        "%{prj.location}/src/win32_time.h",
        "%{prj.location}/src/win32_time.c",
        "%{prj.location}/src/win32_thread.h",
        "%{prj.location}/src/win32_thread.c",
        "%{prj.location}/src/win32_window.c",
        "%{prj.location}/src/wgl_context.c",
        "%{prj.location}/src/egl_context.c",
        "%{prj.location}/src/osmesa_context.c"
		}

		defines 
		{ 
			"_GLFW_WIN32",
			"_CRT_SECURE_NO_WARNINGS"
		}
	
	filter "system:linux"
		files
		{
			"%{prj.location}/src/glx_context.c",
			"%{prj.location}/src/glx_context.h",
			"%{prj.location}/src/linux_joystick.c",
			"%{prj.location}/src/linux_joystick.h",
			"%{prj.location}/src/posix_time.c",
			"%{prj.location}/src/posix_time.h",
			"%{prj.location}/src/posix_thread.c",
			"%{prj.location}/src/posix_thread.h",
			"%{prj.location}/src/x11_init.c",
			"%{prj.location}/src/x11_monitor.c",
			"%{prj.location}/src/x11_platform.h",
			"%{prj.location}/src/x11_window.c",
			"%{prj.location}/src/xkb_unicode.c",
			"%{prj.location}/src/xkb_unicode.h"
		}

		defines 
		{ 
			"_GLFW_X11"
		}
		
	filter "system:macosx"
		files
		{
			"%{prj.location}/src/cocoa_init.m",
			"%{prj.location}/src/cocoa_joystick.m",
			"%{prj.location}/src/cocoa_joystick.h",
			"%{prj.location}/src/cocoa_monitor.m",
			"%{prj.location}/src/cocoa_platform.h",
			"%{prj.location}/src/cocoa_time.c",
			"%{prj.location}/src/cocoa_window.m",
			"%{prj.location}/src/nsgl_context.m",
			"%{prj.location}/src/nsgl_context.h",
			"%{prj.location}/src/posix_thread.c",
			"%{prj.location}/src/posix_thread.h"
		}

		defines
		{ 
			"_GLFW_COCOA"
		}