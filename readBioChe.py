import pandas as pd

def read_excel_with_pandas(file_path, sheet_name=0):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        print("文件读取成功！前5行数据：")
        cha = df.iloc[4]['AST']
        df = df.replace(cha, 0)
        df = df.replace('M2', 0)
        df = df.replace('M5', 1)
        return df
    except Exception as e:
        print(f"读取失败: {e}")
        return None
    
def getSomeCols(df, cols):
    new_df = df.loc[:, cols]
    return new_df

if __name__ == '__main__':
    # 示例调用
    file_path = "Data/M2M5BIOCHE.xlsx"  # 替换为实际文件路径
    data = read_excel_with_pandas(file_path, sheet_name='Sheet2')
    data = getSomeCols(data, cols=['ALT', 'AST', '总胆红素', '白蛋白', '球蛋白', '肌酐', 'Ccr', 'K', 'Ca', 'P', 'Na', '尿酸', '甘油三酯', '胆固醇', 'LDL', 'HDL', '尿素', '分型'])
    print(data.head())