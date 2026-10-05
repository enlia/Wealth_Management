"""获取本机硬件信息（不依赖 wmic/PowerShell，用 ctypes 调 Win32 API）"""
import ctypes
import platform
import os
import time

# 内存
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

m = MEMORYSTATUSEX()
m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
GB = 1024 ** 3
print("=" * 56)
print(f"系统      : {platform.system()} {platform.release()} ({platform.version()})")
print(f"CPU       : {platform.processor() or '见下'}  逻辑核心 {os.cpu_count()}")
print(f"内存总量  : {m.ullTotalPhys/GB:.1f} GB")
print(f"内存可用  : {m.ullAvailPhys/GB:.1f} GB   (已用 {m.dwMemoryLoad}%)")
print(f"Python    : {platform.python_version()}  ({platform.architecture()[0]})")

# 显卡：读注册表
try:
    import winreg
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                          r"SYSTEM\CurrentControlSet\Control\Video")
    names = set()
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(key, i); i += 1
        except OSError:
            break
        try:
            k2 = winreg.OpenKey(key, sub)
            val, _ = winreg.QueryValueEx(k2, "DriverDesc")
            names.add(str(val))
        except OSError:
            pass
    for n in sorted(names):
        print(f"显卡      : {n}")
except Exception as e:
    print("显卡      : 读取失败", e)

# 磁盘
try:
    import shutil
    t, u, f = shutil.disk_usage("C:\\")
    print(f"C盘       : 总 {t/GB:.0f} GB | 已用 {(t-u)/GB:.0f} GB | 剩余 {u/GB:.0f} GB")
except Exception as e:
    print("磁盘      :", e)
print("=" * 56)
