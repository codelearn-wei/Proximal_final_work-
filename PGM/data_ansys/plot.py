import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_csv(file_path, save_folder):
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    
    # 设置图表风格
    plt.style.use('seaborn-darkgrid')  # 使用 seaborn 的暗网格风格
    
    # 绘制图表
    plt.figure(figsize=(12, 7))
    plt.plot(data['Iteration'], data['GradientNorm'], marker='o', markersize=4, linestyle='-', color='royalblue', linewidth=2)
    plt.title('Gradient Norm vs. Iteration', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Gradient Norm', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 增加图例（如果有多条线或需要图例的话）
    plt.legend(['Gradient Norm'], fontsize=12)
    
    # 获取文件名用于保存图像
    filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(filename)[0]
    save_path = os.path.join(save_folder, filename_no_ext + '.png')
    
    # 保存图表
    plt.savefig(save_path, format='png', dpi=300)  # 增加dpi参数提高图像质量
    plt.close()  # 关闭图表以释放内存

def main():
    # 指定 CSV 文件所在的文件夹和保存图表的文件夹
    csv_folder = 'PGM/data_ansys/grad_data'  # 更新为您的 CSV 文件夹路径
    save_folder = 'PGM/data_ansys/graph'  # 更新为您希望保存图表的文件夹路径
    
    # 确保保存图表的文件夹存在
    os.makedirs(save_folder, exist_ok=True)
    
    # 遍历文件夹中的所有 CSV 文件
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(csv_folder, file)
            plot_from_csv(file_path, save_folder)

if __name__ == '__main__':
    main()
