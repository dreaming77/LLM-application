import csv


def read_csv_first_10_lines(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', newline='', encoding=encoding) as f:
            # 创建CSV读取器（默认分隔符为逗号，如需其他分隔符可指定delimiter参数）
            csv_reader = csv.reader(f)

            # 读取表头（如果有）
            header = next(csv_reader, None)
            if header:
                print("表头：", header)
                print("-" * 50)

            # 读取前十行数据（若包含表头，这里实际读取9行数据，可根据需求调整）
            for i, row in enumerate(csv_reader, start=1):
                if i > 10:
                    break
                print(f"第{i}行：", row)
                print("-" * 50)

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except UnicodeDecodeError:
        print(f"错误：编码错误，请尝试修改encoding参数（如'gbk'）")
    except Exception as e:
        print(f"发生错误：{e}")


# 使用示例
if __name__ == "__main__":
    csv_file_path = "../datasets/Chinese-medical-dialogue-data/Data/Andriatria/Andriatria.csv"
    read_csv_first_10_lines(csv_file_path, encoding='gb2312')
