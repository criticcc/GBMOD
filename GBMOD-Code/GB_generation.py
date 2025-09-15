import time

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.special import gamma
import warnings
warnings.filterwarnings("ignore")


def calculate_center_radius(gb):

    """
     作用：计算给定粒球的几何中心和最大半径（中心到各点的最大欧式距离）。
输入：gb: np.ndarray，形状 (p, m)。
输出：

center: np.ndarray，形状 (m, )；

radius: float，标量；
实现要点：radius = max(‖x_i - center‖_2)。
    Args:
        gb:

    Returns:

    """
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDistances = np.sum(diffMat ** 2, axis=1)
    radius = max(sqDistances ** 0.5)
    return center, radius



def division_central_consistency(gb_list, gb_list_not):
    '''
作用：对一批粒球用“中心一致性”判据决定是否二分。

“中心一致性（ccp）”见 get_ccp，粗意为“平均半径内密度 / 最大半径内密度”。

    '''
    gb_list_new = []
    methods = '2-means' #methods 固定为 '2-means'（在 spilt_ball_k_means 里走 n_init=1、max_iter=2 的极速二分）。
    for gb in gb_list:
        if len(gb) > 1:#只有样本数 p>1 才可能二分；单点或空球直接判为“不分裂”。
            ball_1, ball_2 = spilt_ball_k_means(gb, 2, methods)#用 2-means 把 gb 二分。返回两个子球：
            #防御式检查：若 k-means 出现空簇（退化情况），放弃分裂，原球并入 gb_list_not，处理下一个。
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_not.append(gb)
                continue

            """计算中心一致性及其分裂建议：
            ccp = 密度(平均半径内) / 密度(最大半径内)；     
            ccp_flag = (ccp >= 1.30) or (ccp < 1)：为 True 时倾向分裂。"""
            ccp, ccp_flag = get_ccp(gb)
            _, radius = calculate_center_radius(gb)
            #计算父球半径（这里的 radius 没被后续使用，是冗余的；可删）。


            """
            计算“稀疏度”指标（命名应为 sparse_* 更合适）：
            对样本数 >2 的粒球：返回 p / radius^m（越大越稠密、越小越稀疏）；            
            对样本数 ≤2 的粒球：直接返回 半径（单位不同，是启发式便携写法，供边界特判使用）。"""
            sprase_parent = get_dm_sparse(gb)
            sprase_child1 = get_dm_sparse(ball_1)
            sprase_child2 = get_dm_sparse(ball_2)



            t1 = ccp_flag #t1：满足中心一致性分裂建议（必要条件之一）。
            t4 = len(ball_2) > 2 and len(ball_1) > 2
            """
            t4：规模约束——两侧子球都要有至少 3 个点，以避免产生极小簇。
            这是一条常规的“健康子簇”约束。"""


            if (sprase_child1 >= sprase_parent or sprase_child2 >= sprase_parent) and (
                    len(ball_1) == 1 or len(ball_2) == 1):
                t4 = True
            """关键特判（例外放行）：
            如果出现单点子球（len==1），按规模约束本应不允许分裂；            
            但若该单点子球的“稀疏度指标”不低于父球（sprase_child >= sprase_parent），则放宽规模约束（t4=True），允许把这“单点簇”剥离出来。           
            直觉解释：           
            get_dm_sparse 对 len<=2 返回半径；半径大往往意味着这个点离中心/其它点很远（像潜在异常/离群“尖刺”），           
            sprase_child >= sprase_parent（注意这里对单点的度量是半径）意味着“这个单点并不比父球更‘紧凑’，甚至更‘离散’/远”，因此让它作为独立子球是合理的（把“远离的点”剥出来）。           
            注意：这里比较的是混合量纲（父球是密度、单点是半径），纯粹是经验启发式，但确实能抓出“很远的单点”。"""

            if t1 and t4:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
            """
            t1（一致性建议）与 t4（规模/例外规则）同时成立 → 接受分裂，两个子球加入新候选池；
            否则 → 不分裂，原球加入已确定池。"""
        else:
            gb_list_not.append(gb)

    return gb_list_new, gb_list_not


def division_central_consistency_strong(gb_list,gb_list_not):
    '''
作用：与上类似，但使用更强的判据 get_ccp_strong（四分之一最大半径内密度与最大半径内密度之比）。
输入/输出：同 division_central_consistency。
内部规则：当 ccp_flag_strong 为真且两个子球都 > 2 才分裂，否则记为“不分裂”。

    '''
    gb_list_new = []
    methods = '2-means'

    for gb in gb_list:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball_k_means(gb, 2, methods)
            ccp, ccp_flag = get_ccp_strong(gb)
            t1 = ccp_flag
            t4 = len(ball_2) > 2 and len(ball_1) > 2
            if t1 and t4:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)

    return gb_list_new,gb_list_not


def spilt_ball_k_means(data, n, methods):
    """
    作用：用 KMeans 将 data 分为 n 个子簇并返回为“子球”。
    输入：

    data: np.ndarray，形状 (p, m)；

    n: int（簇数）；

    methods: str，支持 '2-means'（固定 n=2，n_init=1, max_iter=2）或 'k-means'（一般 KMeans）。
    输出：

    balls: list[np.ndarray]，长度 = n，第 j 个的形状 (p_j, m)，且 ∑ p_j = p。
    备注：

    '2-means' 分支里强行返回两个子球 [ball1, ball2]。

    max_iter=2 很小，只做非常粗糙的二分；若要稳定可适当增大。
    Args:
        data:
        n:
        methods:

    Returns:

    """


    if methods == '2-means':
        
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=1, max_iter=2)
        kmeans.fit(data)
        labels = kmeans.labels_
        cluster1 = [data[i].tolist() for i in range(len(data)) if labels[i] == 0]
        cluster2 = [data[i].tolist() for i in range(len(data)) if labels[i] == 1]
        ball1 = np.array(cluster1)
        ball2 = np.array(cluster2)
        return [ball1, ball2]

    elif methods == 'k-means':

        kmeans = KMeans(n_clusters=n, random_state=1)
        kmeans.fit(data)

        
        labels = kmeans.labels_

        
        clusters = [[] for _ in range(n)]
        for i in range(len(data)):
            for cluster_index in range(n):
                if labels[i] == cluster_index:
                    clusters[cluster_index].append(data[i].tolist())

        balls = [np.array(cluster) for cluster in clusters]
        
        
        return balls
    else:
        pass


def divide_gb_k(data, k):
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data)
    labels = kmeans.labels_
    gb_list_temp = []
    for idx in range(k):
        cluster1 = [data[i].tolist() for i in range(len(data)) if labels[i] == idx]
        gb_list_temp.append(np.array(cluster1))
    return gb_list_temp



def spilt_ball(data):
    """
    几何二分，这是最后一步用的分法，和中间粒球分裂的二分不一样
    想象在一个粒球里找出最远的两个点，它们可以看作“这个球的两个对立端”。
然后让其他点“选边站”：靠近哪边就归属哪边。
这样就把原粒球自然地分成了两个子簇。
    Args:
        data:

    Returns:

    """
    ball1 = []
    ball2 = []
    A = pdist(data)
    d_mat = squareform(A)
    r, c = np.where(d_mat == np.max(d_mat))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if d_mat[j, r1] < d_mat[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])

    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]

def get_ccp(gb):
    '''
    得到中心一致性，中心一致性指平均半径内的样本密度与最大半径内的样本密度的比值，如果比值在1~1.3就不分裂，
    否则分裂为两个粒球。密度指：样本个数 / 半径 ** 维度。
    Args:
        gb:

    Returns:

    '''
    num = len(gb)#如果粒球里没有点，直接返回 (0, False)，即 ccp=0，不分裂。
    if num == 0:
        return 0, False
    
    dimension = len(gb[0])
    center = gb.mean(axis=0)#粒球的几何中心（所有点的均值向量），形状 (m,)
    
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5  #计算每个点到中心的欧氏距离向量 distances (p,)

    avg_radius = np.mean(distances) #平均半径与最大半径。
    max_radius = np.max(distances)

    """
    平均半径内密度：用“内点数 / 球体体积近似(
    rˉd
    r
    ˉ
    d
    )”。
    
    最大半径内密度：用“总点数 / 
    Rd
    R
    d
    ”。"""
    points_inside_avg_radius = np.sum(distances <= avg_radius)## 标量，内点个数
    density_inside_avg_radius = points_inside_avg_radius / (avg_radius ** dimension)
    density_max_radius = num / (max_radius ** dimension)
    ccp = density_inside_avg_radius / density_max_radius
    ccp_flag = ccp >= 1.30 or ccp < 1 # 分裂
    """
    得到 ccp；若过大（中心更拥挤，可能多模/应细分）或过小（中心更稀，像“环形/空心”结构），都建议分裂；只有在 [1,1.3) 内视为“中心一致性良好”，不分裂。"""
    return ccp, ccp_flag

def get_ccp_strong(gb):
    '''
    强中心一致性，最大半径的四分之一半径内密度占最大半径内密度的比值
    强一致性 (ccp_strong)：改用更小的半径 (R/4)，更敏感地检测“中心是否过拥挤/过空”。

    结果：强一致性更苛刻，能更快把结构复杂的粒球分裂开来。
    Args:
        gb:

    Returns:

    '''
    num = len(gb)
    if num == 0:
        return 0, False
    dimension = len(gb[0])
    center = gb.mean(axis=0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    max_radius = np.max(distances)
    quarter_radius = max_radius / 4
    points_inside_quarter_radius = np.sum(distances <= quarter_radius)
    density_inside_quarter_radius = points_inside_quarter_radius / (quarter_radius ** dimension)
    density_max_radius = num / (max_radius ** dimension)
    ccp_strong = density_inside_quarter_radius / density_max_radius
    ccp_flag_strong = ccp_strong >= 1.3 or ccp_strong < 1
    return ccp_strong, ccp_flag_strong


def get_dm_sparse(gb):
    num = len(gb)
    dim = len(gb[0])

    if num == 0:     # 如果粒球里没有点
        return 0     # 稀疏度定义为 0
    center = gb.mean(0)  # 粒球中心：对每个维度求平均
    diff_mat = center - gb  # 每个样本点与中心的差向量
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5   # 再开方，得到欧式距离
    radius = max(distances)  # 半径 = 到中心最远的点的距离

    sparsity = num / (radius ** dim)# 稀疏度定义：点数 / 半径^维度
                                       # 可以理解为“单位体积里的点数密度”
    
    

    if num > 2:           # 如果点数大于 2
        return sparsity   # 返回稀疏度
    else:
        return radius      # 如果点太少（<=2），就直接返回半径
        
def get_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius

def de_sparse(gb_list):
    """get_radius(x)：计算粒球 x 的半径（中心到最远点的距离）。
    get_radius(x) / len(x)：即“平均每个点对应的半径尺度”，可以理解为一种半径-点数比值，值越大，说明球既大又点数少 → 稀疏。
    对所有粒球求平均，得到整体的参考值 avg_r_div_n。"""
    avg_r_div_n = sum([get_radius(x) / len(x) for x in gb_list]) / len(gb_list)
    """所有粒球半径的平均值 avg_r。
    这是另一种全局参考指标，衡量“平均粒球大小”。"""
    avg_r =sum([get_radius(x)  for x in gb_list]) / len(gb_list)
    gb_list_new = []
    gb_split_list_new = []
    """
    gb_list_new：存放“保留的粒球”或“分裂后的子球”。
    gb_split_list_new：存放“候选需要再分裂的粒球”。"""
    for gb in gb_list:
        
        r_t = get_radius(gb)
        if r_t  / len(gb) > avg_r_div_n and r_t > avg_r:
            gb_split_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    """
    r_t / len(gb) > avg_r_div_n：如果它的半径-点数比值比全局平均还大，说明它比一般粒球更稀疏；
    r_t > avg_r：并且它的半径大于平均半径，说明它比一般粒球更大。
    两个条件都满足 → 判定为过稀疏，放进 gb_split_list_new，准备再分裂。 
    否则，直接保留到 gb_list_new。"""

    for gb in gb_split_list_new:#对所有“稀疏待分裂粒球”再做一次几何二分（spilt_ball）：
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball(gb)
            gb_list_new.extend([ball_1, ball_2])
        else:
            gb_list_new.append(gb)

    return gb_list_new

def GB_Gen(data,plt_flag=False):
    '''
    根据数据集生成粒球划分
        x
        Args:
            data:
        Returns:
    '''

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit_transform(data)
    gb_list_not_temp = []
    k1 = int(np.sqrt(len(data)))#计算初始聚类数 k1 = ⌊√n⌋。
    gb_list_temp = divide_gb_k(data, k1) # 先粗划分为根号n个粒球，gb_list_temp: list[np.ndarray]，长度 k1，每个子球形状 (n_j, m)，∑ n_j = n。
    
    gb_list_temp,gb_list_not_temp = division_central_consistency_strong(gb_list_temp,gb_list_not_temp)
    """对“待细分”集合做一轮强中心一致性判据：

    能分就二分 → 生成子球进入新的 gb_list_temp；
    
    不能分的 → 进入 gb_list_not_temp（累积）。
    
    两个返回值类型均为 list[np.ndarray]"""


    gb_list_temp = gb_list_temp + gb_list_not_temp
    gb_list_not_temp = []

    """
    把“强一致性”这一轮的全部结果（无论能否分裂）都放回到 gb_list_temp，作为下一阶段继续尝试的起点；

    清空“未分裂池”。
    
    目的：强判据只做一轮，然后统一进入“普通一致性”循环继续细化。"""

    i = 0
    # 根据中心一致性进行粒球细分
    while 1:
        #进入主循环，用普通中心一致性不断尝试细分。
        i += 1
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        #ball_number_old: int，记录本轮尝试前“已知粒球总数”（= 待细分 + 已确定）。
        gb_list_temp, gb_list_not_temp = division_central_consistency(gb_list_temp, gb_list_not_temp)
        """
        对“待细分集合 gb_list_temp”逐个判定：
        能分裂的 → 其子球进入新的 gb_list_temp（下一轮的“待细分集合”）；
        不能分裂的 → 进入（累积）gb_list_not_temp。    
        循环不变式（非常关键）：    
        调用前：gb_list_temp = 本轮候选；gb_list_not_temp = 之前所有已确定。   
        调用后：gb_list_temp = 本轮新产生的子球（下轮待分）；gb_list_not_temp = 之前已确定 + 本轮不再分的那些。
        """
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp) #计算这一轮调用后的总粒球数。

        if ball_number_new == ball_number_old: #如果总数不再增长，说明这一轮没人被分裂：
            gb_list_temp = gb_list_not_temp
            break

    count = 0
    for gb in gb_list_temp:
        if len(gb) == 1:
            pass
        else:
            count += 1
    #统计非单点粒球的数量（count 未被使用，属于调试/遗留代码，可删）



    gb_list_temp = [x for x in gb_list_temp if len(x) != 0] #清理空粒球（例如极端 KMeans/二分退化可能出现）。
    
    gb_list_temp = de_sparse(gb_list_temp)
    #去稀疏：对半径过大/密度过低的粒球再做一次几何二分（spilt_ball），以便让划分更均衡。
    return gb_list_temp


def load_txt_dataset(path):
    
    with open(path, 'r') as file:
        lines = file.readlines()
    data = np.array([[float(num) for num in line.split()] for line in lines])
    return data
