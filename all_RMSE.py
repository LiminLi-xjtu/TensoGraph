import math
import numpy as np
import pandas as pd

a1 = []
a2 = []
a3 = []
a4 = []
a5 = []
a6 = []
a7 = []
for i in range(10):
    '''
    #oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_svr2-1/results_loewe/'
                       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_svr2-1/results_bliss/'
       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_svr2-1/results_hsa/'
       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_svr2-1/results_zip/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a1.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np1 = np.array(a1, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b1= np.mean(a_np1, axis=0)
for i in range(10):
    '''
    # oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_forest/results_loewe/'
                       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_forest/results_bliss/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_forest/results_hsa/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_forest/results_zip/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a2.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np2 = np.array(a2, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b2= np.mean(a_np2, axis=0)
for i in range(10):
    '''
    # oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_xgboost/results_loewe/'
                      'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_xgboost/results_bliss/'
                   'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_xgboost/results_hsa/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results1_xgboost/results_zip/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a3.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np3 = np.array(a3, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b3= np.mean(a_np3, axis=0)
for i in range(10):
    '''
    # oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/math_oneil_loewe/results_loewe/'
                      'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/math_bliss/results_bliss/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/math_hsa/results_hsa/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/math_zip/results_zip/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a4.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np4 = np.array(a4, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b4= np.mean(a_np4, axis=0)
for i in range(10):
    '''
    # oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/output/cv_cv_oneil_loewe2404042334/'
                       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/output/cv_oneil_bliss2403301927/'
                       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/output/cv_oneil_hsa2403302334/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/output/cv_oneil_zip2403302338/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a5.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np5 = np.array(a5, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b5= np.mean(a_np5, axis=0)
for i in range(10):
    '''
    # oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/zdx/results/results_loewe/results_oneil_mgaedc100_folds/'
                      'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results/results_bliss/results_oneil_mgaedc100_folds/'
                      'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results/results_hsa/results_oneil_mgaedc100_folds/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/results/results_zip/results_oneil_mgaedc100_folds/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a6.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np6 = np.array(a6, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b6= np.mean(a_np6, axis=0)
for i in range(10):
    '''
    # oneil loewe
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/result_dgt/results_oneil_loewe/results_oneil_mgaedc100_folds/'
                       'test_data_cells_stats_'+str(i)+'.txt', sep='\t', header=None)
    
    # oneil bliss
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/result_dgt/results_oneil_bliss/results_oneil_mgaedc100_folds/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    
    # oneil hsa
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/result_dgt/results_oneil_hsa/results_oneil_mgaedc100_folds/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)
    '''
    # oneil zip
    data = pd.read_csv('/media/imin/DATA/zhangdongxue/result_dgt/results_oneil_zip/results_oneil_mgaedc100_folds/'
                       'test_data_cells_stats_' + str(i) + '.txt', sep='\t', header=None)

    sqrt_values= (data.iloc[:, 0]).apply(math.sqrt)  # 对第一列每个元素应用开平方根函数
    # 将结果添加到列表 a 中
    a7.append(sqrt_values.tolist())  # 转换为列表并添加
# 转换为 NumPy 数组
a_np7 = np.array(a7, dtype=float)  # 假设数据是浮点数类型
# 计算每列数据的均值
b7= np.mean(a_np7, axis=0)
# 定义细胞系列表
cell_lines = [
    'A2058', 'A2780', 'A375', 'A427', 'CAOV3', 'COLO320DM', 'DLD1', 'EFM192B', 'ES2', 'HCT116',
    'HT144', 'HT29', 'KPL1', 'LNCAP', 'LOVO', 'MDAMB436', 'MSTO', 'NCIH1650', 'NCIH2122', 'NCIH23',
    'NCIH460', 'NCIH520', 'OCUBM', 'OV90', 'OVCAR3', 'PA1', 'RKO', 'RPMI7951', 'SKMEL30', 'SKMES1',
    'SKOV3', 'SW620', 'SW837', 'T47D', 'UACC62', 'UWB1289', 'UWB1289BRCA1', 'VCAP', 'ZR751'
]
# 将 b1 到 b7 转换为 DataFrame，并转置
df = pd.DataFrame({
    'SVR': b1,
    'Random Forest': b2,
    'XGBoost': b3,
    'Matchmaker': b4,
    'PRODeepSyn': b5,
    'MGAE-DC': b6,
    'OURS': b7
}, index=cell_lines)
# 在 DataFrame 最前面加入 'cell' 列
df.insert(0, 'cell', df.index)
# 按照 'OURS' 列从大到小排序
df1 = df.sort_values(by='OURS', ascending= True)
#print(df1)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 获取方法名称，即除索引外的所有列名
methods = ['SVR','Random Forest','XGBoost', 'Matchmaker','PRODeepSyn','MGAE-DC','OURS']

# 绘制盒子图
plt.figure(figsize=(7, 5.5))  # 设置图形大小

# 使用Seaborn绘制盒子图
ax1=sns.boxplot(data=df1[methods])
ax1.set_xticklabels([])

# 添加图例，包括带有与盒图相对应的颜色的长方形以及SVR、Random Forest、XGBoost、Matchmaker、PRODeepSyn、MGAE-DC和OURS
legend_labels = ['SVR', 'Random Forest', 'XGBoost', 'Matchmaker', 'PRODeepSyn', 'MGAE-DC', 'OURS']
colors = sns.color_palette()[:len(legend_labels)]
legend_handles = [plt.Rectangle((0,0),0.3,0.3, color=colors[i], alpha=0.6) for i in range(len(legend_labels))]
ax1.legend(legend_handles, legend_labels, loc='best', fontsize=12)
#plt.title('RMSE',fontsize=36)
# 设置各轴的粗细
'''
plt.gca().spines['top'].set_linewidth(4)     # 上轴
plt.gca().spines['bottom'].set_linewidth(4)  # 下轴
plt.gca().spines['left'].set_linewidth(4)    # 左轴
plt.gca().spines['right'].set_linewidth(4)   # 右轴
'''
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('RMSE',fontsize=19,fontweight='bold')
# 使用 tight_layout() 自适应画布
plt.tight_layout()
# 设置标题
#plt.title('RMSE Box Plot for Different Methods across Cell Lines')
plt.savefig('/home/zhangdongxue/all_RMSE/oneil/ALL_RMSE_oneil_zip')

# 显示图形
plt.show()

# 如果你希望保存图形为图片文件，可以使用下面的代码
# plt.savefig('rmse_boxplot_across_cell_lines.png')