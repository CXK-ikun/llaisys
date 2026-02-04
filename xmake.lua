add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- 统一定义 Windows 平台的兼容性配置函数
function add_windows_flags()
    if is_plat("windows") then
        add_cxflags("/WX-")            -- 关闭“将警告视为错误”
        add_cxflags("/wd4267")         -- 屏蔽 size_t 转 int 的警告 (tensor.cpp 报错项)
        add_cxflags("/wd4244")         -- 屏蔽 double 转 float 的精度丢失警告
        add_cxflags("/utf-8")          -- 强制使用 UTF-8 编码
        add_defines("_CRT_SECURE_NO_WARNINGS") -- 禁用 Windows 安全函数警告
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
end

target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_windows_flags() -- 使用兼容配置

    add_files("src/utils/*.cpp")
    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_windows_flags() -- 使用兼容配置

    add_files("src/device/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_windows_flags() -- 使用兼容配置

    add_files("src/core/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_windows_flags() -- 使用兼容配置

    add_files("src/tensor/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_windows_flags() -- 使用兼容配置
    
    add_files("src/ops/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_windows_flags() -- 使用兼容配置

    add_files("src/llaisys/*.cpp", "src/llaisys/models/*.cpp")
    set_installdir(".")

    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            -- 修正：Windows 下编译产物通常在 bin/ 目录下且为 .dll
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()