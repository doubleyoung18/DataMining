def load_D(filename):
    '''
    读取数据库
    :param filename:文件名
    :return:
        D:数据库
    '''
    with open(filename, "r") as file:
        D = []
        for line in file:
            transaction = set(line.rstrip('  \r\n').split("  "))
            D.append(transaction)
    return D

def save_L(L):
    with open("support.txt", "w") as file:
        count = 1
        for item_set in L:
            for item_set1 in item_set:
                item = list(map(int, item_set1))
                print(count, ":", item)
                file.write(str(item)+'\n')
                count += 1

def create_C1(D):
    '''
    创建第1轮迭代候选集
    :param D: 数据库
    :return:
        C1：第1轮迭代候选集
    '''
    C1 = set()
    for transaction in D:
        for item in transaction:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1

def is_apriori(Ci_item, Lisub1):
    '''
    根据先验规则，判断是否是频繁项集
    :param Ci_item:第i轮迭代候选集候选项
    :param Lisub1:第i-1轮迭代频繁项集
    :return:
        判断结果
    '''
    for item in Ci_item:
        Ci_sub = Ci_item - frozenset(item)
        if Ci_sub not in Lisub1:
            return False
    return True

def create_Ci(Lisub1, k):
    '''
    创建第i轮迭代候选集
    :param Lisub1:第i-1轮迭代频繁项集
    :param k:第i轮频繁项集项数
    :return:
        Ci:第i轮迭代候选集
    '''
    Ci = set()
    Lisub1_len = len(Lisub1)
    Lisub1_list = list(Lisub1)
    for i in range(Lisub1_len):
        for j in range(1, Lisub1_len):
            list1 = list(Lisub1_list[i])
            list2 = list(Lisub1_list[j])
            list1.sort()
            list2.sort()
            if list1[0:k-2] == list2[0:k-2] and list1 != list2:
                Ci_item = Lisub1_list[i] | Lisub1_list[j]
                if is_apriori(Ci_item, Lisub1):
                    Ci.add(Ci_item)
    return Ci


def generate_Li_by_Ci(D, Ci, min_sup, support_dict):
    '''
    由第i轮迭代候选集生成该轮频繁项集
    :param D:数据库
    :param Ci:第i轮迭代候选集
    :param min_sup:最小支持度
    :param support_dict:频繁项计数字典
    :return:
        Li:第i轮迭代生成的频繁项集
    '''
    Li = set()
    item_dict = {}  # 候选项计数字典
    min_sup_num = float(len(D)) * min_sup  # 最小支持度计数
    # 统计各候选项的支持度计数
    for transaction in D:
        for item in Ci:
            if item.issubset(transaction):  # 若事务中含此项，为此项的支持度计数
                if item not in item_dict:
                    item_dict[item] = 1
                else:
                    item_dict[item] += 1
    # 根据支持度计数生成频繁项集
    for item in item_dict:
        if item_dict[item] >= min_sup_num:
            Li.add(item)
            support_dict[item] = item_dict[item]
    return Li

def generate_L(D, k, min_sup):
    '''
    生成所有的频繁项集
    :param D:数据库
    :param k:频繁项最大项数
    :min_sup:最小支持度
    :return:
        Lmax:所有最大长度的频繁项集
    '''
    support_dict = {}  # 频繁项全集字典
    C1 = create_C1(D)
    L1 = generate_Li_by_Ci(D, C1, min_sup, support_dict)
    Lisub1 = L1.copy()
    Lmax = Lisub1
    for i in range(2, k+1):
        Ci = create_Ci(Lisub1, i)
        Li = generate_Li_by_Ci(D, Ci, min_sup, support_dict)
        if len(Li) == 0:  # 若频繁项集为空集，结束迭代
            break
        Lisub1 = Li.copy()
        Lmax = Lisub1
    return Lmax

if __name__ == "__main__":
    filename = "apriori_data.txt"
    support_dict = {}

    D = load_D(filename)

    Lmax = generate_L(D, 100, 0.05)  # min_support = 100 / 2000 = 0.05
    print("最大频繁项集：")
    for item_set in Lmax:
        item = sorted(list(map(int, item_set)))
        print(item)