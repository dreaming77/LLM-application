import pynvml


def check_gpu_memory():
    # 初始化NVML库
    pynvml.nvmlInit()

    try:
        # 获取GPU数量
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {device_count} 个GPU设备:\n")

        for i in range(device_count):
            # 获取GPU句柄
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # 获取GPU名称
            name = pynvml.nvmlDeviceGetName(handle)

            # 获取内存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # 转换为GB单位（1GB = 1024^3字节）
            total_mem = mem_info.total / (1024 ** 3)
            used_mem = mem_info.used / (1024 ** 3)
            free_mem = mem_info.free / (1024 ** 3)

            # 计算使用率
            usage_percent = (used_mem / total_mem) * 100

            # 打印信息
            print(f"GPU {i}: {name}")
            print(f"  总内存: {total_mem:.2f} GB")
            print(f"  已使用: {used_mem:.2f} GB ({usage_percent:.1f}%)")
            print(f"  空闲内存: {free_mem:.2f} GB")
            print("-" * 50)

    except pynvml.NVMLError as e:
        print(f"NVML错误: {e}")
    finally:
        # 关闭NVML库
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    check_gpu_memory()
