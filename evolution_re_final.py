#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by PyCharm Community Edition
import numpy as np
import re
import copy
import sre_constants
from collections import Counter
import csv
import logging
import os
import json
import sys

"""
用 python 3 运行
缺少numpy库:
    pip install numpy
"""

"""
主要调整部分
- population_num, dim, crossover_rate， mutate_rate。
- '.*', '' 等各所占比例。
- 适应度函数的设置，如何能尽量增大区分度。轮盘赌概率分配。
- 部分匹配与全部匹配的适应度怎么设置，如对于 'hello' ， '.*' 和 'h.*o' 和 'h.*a.*o' 的适应度怎么设置合理。又如何编写适应度函数以实现这种设置。


Todo:
- 为防正则表达式冗余，考虑正则表达式长度，短者优先？
- 正则表达式子串匹配？
"""


class EvolutionRe:
    """
    输入：
        strings: 字符串列表
        labels: 对应的label列表，label为0或1
        elements: 字符串分词列表
    算法目标：
        给出一组正则表达式，能够尽量分开label为1和为0的字符串
    输出：
        valid_patterns: 满足要求的正则表达式
    """

    def __init__(self, population_num=200, dim=20, max_gen=1000, crossover_rate=0.2, mutate_rate=0.1, div=1, dtype=np.uint16):
        """
        :param population_num: 种群大小
        :param dim: 正则表达式最大长度
        :param max_gen: 最大迭代次数
        :param crossover_rate: 杂交比率
        :param mutate_rate: 变异比率
        :param div: 参数，用来控制错误容忍度，越大对错误的容忍度越低。若减小该参数，1类匹配率会高，但同时也会增大0类匹配率
        :param dtype: 所用的编码的dtype
        """
        self.strings = []  # 待测试字符串列表
        self.pattern_elements = []  # 组成正则表达式的元素列表
        self.pattern_elements_len = 0  # 组成正则表达式的元素列表长度
        self.labels = []  # 字符串label列表，要匹配的字符串为1
        self.population_num = population_num  # 种群大小
        self.dim = dim  # 正则表达式的长度(以element计)
        self.max_gen = max_gen  # 最大迭代代数
        self.crossover_rate = crossover_rate  # 杂交比率
        self.mutate_rate = mutate_rate  # 变异比率
        self.div = div
        self.dt = np.dtype(dtype)  # 所用的编码的dtype
        self.dt_max = np.iinfo(self.dt).max  # 编码单元的最大值
        self.dt_min = np.iinfo(self.dt).min  # 编码单元的最小值
        self.max_str_len = 0
        self.pattern_dict = {}  # 记录已计算过适应度的pattern的字典，pattern->适应度的映射
        self.valid_patterns = {}  # 存储可用的正则表达式,pattern->ls的映射，ls为匹配的string的下标构成的列表
        self.eps = 1e-10
        self.l1_set = set()
        self.l0_set = set()
        self.re_reduce = re.compile(r'(\.\*)+')

    def decode(self, code_list):
        """对给定的个体编码进行解码，得到该个体对应的pattern字符串
        :param code_list:
        :return: 解码所得的pattern字符串
        """
        st = "".join([self.pattern_elements[x % self.pattern_elements_len] for x in code_list])
        st, n = self.re_reduce.subn(r'.*', st)
        return st

    def init(self):
        """初始化种群
        :return: 初始化种群矩阵，每行为一个个体的编码
        """
        return np.random.randint(self.dt_min, self.dt_max + 1, (self.population_num, self.dim), self.dt)

    def evaluate_str(self, pattern_str):
        """对给定pattern字符串计算适应度
        :param pattern_str: pattern字符串
        :return: 适应度，浮点数类型
        """
        if pattern_str == '.*' * int(len(pattern_str) / 2):  # 如果该pattern全由.*构成，则返回-1
            return -1
        m0, m1 = 1, 1  # m0为pattern字符串所能匹配的label为0的字符串的个数，m1类似
        m0_len, m1_len = 1, 1  # m0_len为pattern字符串所能匹配的label为0的字符串的总匹配长度，m1类似
        for i in range(len(self.strings)):
            try:
                match_object = re.search(pattern_str, self.strings[i])
                if match_object:
                    match_span = match_object.span()
                    ml = (match_span[1] - match_span[0])  # 匹配长度
                    if self.labels[i] == 1:
                        m1 += 1
                        m1_len += ml
                    else:
                        m0 += 1
                        m0_len += ml
                else:
                    pass
            except sre_constants.error:  # pattern编译错误
                return -1
        return m1_len - self.div * m0_len

    def evaluate_code(self, pattern_code):
        """给定一个编码，获取适应度
        :param pattern_code: 编码
        :return: 适应度，浮点型
        """
        pattern_str = self.decode(pattern_code)
        eva_score = self.evaluate_str(pattern_str)
        # 将该pattern_str添加进pattern_dict
        self.pattern_dict[pattern_str] = eva_score
        return eva_score

    def evaluate_one(self, pattern_code):
        """对于一个个体，获取适应度
        对整个pattern获取适应度，然后获取每个element的适应度，将以上所有适应度相加作为该pattern的最终适应度
        :param pattern_code:个体编码
        :return: 适应度，浮点型
        """
        pattern_str = self.decode(pattern_code)
        # 如果该pattern全由.*构成，则返回-1
        if pattern_str == '.*' * int(len(pattern_str) / 2):
            return -1, len(pattern_str)
        if pattern_str in self.pattern_dict:
            return self.pattern_dict[pattern_str], len(pattern_str)
        eva_total_score = 0
        # for code in pattern_code:
        #     eva_total_score += self.evaluate_code([code])
        eva_total_score += self.evaluate_code(pattern_code)
        # eva_total_score -= len(pattern_str) / 1

        # eva_total_score /= self.max_str_len
        # eva_total_score = np.exp(eva_total_score)
        self.pattern_dict[pattern_str] = eva_total_score
        return eva_total_score, len(pattern_str)

    def evaluate(self, p):
        """获取适应度列表，与种群顺序一一对应
        :param p: 种群矩阵
        :return: 适应度列表
        """
        es = [self.evaluate_one(x) for x in p]
        em = min([e[0] for e in es])
        es = [(e[0] - em, e[1]) for e in es]  # 使es都为正值
        # 以下惩罚较长的pattern，对于适应度相同的pattern，使越长的适应度越低，但要保持原适应度的序
        # 如当前适应度为[1, 1, 4, 4, 4, 5]
        # 对应pattern长度分别为[30, 20, 30, 20, 10, 40]
        # 使惩罚后为[1, 2.5, 4, 4.33, 4.66, 5]
        es = [[i, es[i][0], es[i][1]] for i in range(len(es))]
        es = sorted(es, key=lambda a: a[1])
        es.append([-1, es[-1][1] * 1.1 + self.max_str_len, 0])
        es_sort_by_e = []
        le = -1
        for e in es:
            if e[1] != le:
                le = e[1]
                es_sort_by_e.append([])
            es_sort_by_e[-1].append(e)
        for i in range(len(es_sort_by_e)):
            # 适应度相同的按pattern长度从大到小排序
            es_sort_by_e[i] = sorted(es_sort_by_e[i], key=lambda a: a[2], reverse=True)
        for i in range(len(es_sort_by_e) - 1):
            diff = es_sort_by_e[i + 1][0][1] - es_sort_by_e[i][0][1]
            diff /= len(es_sort_by_e[i])
            for j in range(len(es_sort_by_e[i])):
                es_sort_by_e[i][j][1] += diff * j
        new_es = []
        for e in es_sort_by_e:
            for ee in e:
                new_es.append(ee)
        new_es = sorted(new_es, key=lambda a: a[0])
        new_es = [e[1] for e in new_es]
        es = new_es[1:]
        # 惩罚部分结束
        return es

    def parent_select(self, p):
        """父体选择
        :param p: 种群矩阵
        :return: 父体下标构成的列表
        """
        r = np.random.randint(0, len(p), int(self.crossover_rate * len(p)))
        return r.tolist()

    def crossover_one(self, p, p1_i, p2_i, rpos):
        """对一对父代个体在特定位置杂交，获得子代个体
        :param p: 种群列表
        :param p1_i: 父体1的下标
        :param p2_i: 父体2的下标
        :param rpos: 杂交位置
        :return: 杂交获得两个子代个体
        """
        pos = int(rpos / (self.dt.itemsize * 8)) + 1  # 在第几个编码元素上进行杂交
        byte_pos = rpos % (self.dt.itemsize * 8)  # 在该编码元素上第几个二进制位进行杂交

        o1 = copy.deepcopy(p[p1_i])
        o2 = copy.deepcopy(p[p2_i])
        o3 = copy.deepcopy(p[p1_i])  # 作为临时变量

        o1[pos:] = o2[pos:]
        o2[pos:] = o3[pos:]
        pos -= 1
        mask1 = 2**(self.dt.itemsize * 8 - byte_pos) - 1  # 与操作的mask
        mask2 = self.dt_max - mask1  # 与操作的mask
        o1[pos] = (o1[pos] & mask2) + (o2[pos] & mask1)
        o2[pos] = (o2[pos] & mask2) + (o3[pos] & mask1)
        return [o1, o2]

    def crossover(self, p, parents_index):
        """杂交
        :param p: 种群矩阵
        :param parents_index: 父体下标构成的列表
        :return: 杂交获得的所有的子代个体
        """
        rp = np.random.randint(0, len(parents_index), (len(parents_index), 2))  # 随机生成父体下标对
        r_pos = np.random.randint(0, self.dim * self.dt.itemsize * 8, (len(parents_index)))  # 随机生成杂交位置
        children = []
        for i in range(rp.shape[0]):
            children.extend(self.crossover_one(p, parents_index[rp[i][0]], parents_index[rp[i][1]], r_pos[i]))
        return np.array(children)

    def mutate(self, p):
        """变异
        :param p: 种群矩阵
        :return: 变异后的个体
        """
        r = np.random.randint(0, p.shape[0], int(p.shape[0] * self.mutate_rate))  # 随机生成要变异的个体下标
        rr = np.random.randint(0, self.dim, int(p.shape[0] * self.mutate_rate))  # 随机生成要变异的编码元素位置
        # 随机生成在以上编码元素上要变异的二进制位置
        rrr = np.random.randint(0, self.dt.itemsize * 8, int(p.shape[0] * self.mutate_rate))
        mutate_ones = []
        for i in range(len(r)):
            mutate_one = copy.deepcopy(p[r[i]])
            mutate_one[rr[i]] += (2**rrr[i])
            mutate_ones.append(mutate_one)
        return np.array(mutate_ones)

    def test_valid(self):
        """ 测试当前的valid_patterns是否已经满足要求
        :return:  bool值
        """
        ls = []
        for pattern_str in self.valid_patterns:
            pc = re.compile(r'^.*' + pattern_str + r'.*$')
            for i in range(len(self.strings)):
                try:
                    if pc.match(self.strings[i]):
                        ls.append(i)
                except sre_constants.error:
                    return False
        ls1 = set(ls)
        ls0 = set(range(len(self.labels))) - ls1
        tp = len(ls1 & self.l1_set)
        tn = len(ls0 & self.l0_set)
        fp = len(ls1 & self.l0_set)
        fn = len(ls0 & self.l1_set)
        tpr = tp / (tp + fn + self.eps)
        fpr = fp / (fp + tn + self.eps)
        return tpr, fpr

    def get_match_results(self, pattern_str):
        """对一个pattern字符串，获取其匹配的string的下标构成的列表
        :param pattern_str:
        :return: 下标列表
        """
        ls = []
        pc = re.compile(r'^' + pattern_str + r'$')
        for i in range(len(self.strings)):
            try:
                if pc.match(self.strings[i]):
                    ls.append(i)
            except sre_constants.error:
                return False
        return ls

    def sample_p(self, p, es):
        """通过轮盘赌选择，保留种群中的部分个体，使种群数量为population_num
        :param p: 种群矩阵
        :param es: 适应度列表
        :return: 经过选择后的种群矩阵和适应度列表
        """
        p_dict = {self.decode(p[i]): i for i in range(len(p))}
        # 去除pattern相同而编码不同的重复个体
        p = np.array([p[v] for k, v in p_dict.items()])
        es = np.array([es[v] for k, v in p_dict.items()])
        ess = es
        es_sum = []
        sum_temp = 0
        for e in ess:
            es_sum.append(sum_temp)
            sum_temp += e
        es_sum.append(sum_temp + 1)
        rd = np.random.randint(0, sum_temp + 1, self.population_num)
        reserve_indexes = []
        for r in rd.tolist():
            for i in range(len(es_sum) - 1):
                if es_sum[i] <= r < es_sum[i + 1]:
                    reserve_indexes.append(i)
                    break
        p = np.array([p[x] for x in reserve_indexes])
        es = [es[x] for x in reserve_indexes]
        return p, es

    def get_pattern_elements(self, elements, stop_words_set, first_n):
        """给定分词列表，获取其中可用于作为正则表达式元素的词语列表
        :param elements: 分词列表
        :param stop_words_set: 停用词集合
        :param first_n: 取在两类字符串列表中频率差最大的前first_n个词语
        :return: 可用于作为正则表达式元素的词语列表
        """
        elements_1 = [elements[i] for i in range(
            len(self.labels)) if self.labels[i]]  # 属于label 1的分词的列表
        elements_0 = [elements[i] for i in range(
            len(self.labels)) if self.labels[i] == 0]  # 属于label 0的分词的列表
        ele_1, ele_0 = [], []  # ele_1为label 1中的所有词语，ele_0为label 0中的所有词语
        for x in elements_1:
            ele_1.extend(x)
        for x in elements_0:
            ele_0.extend(x)
        ele_1 = [e for e in ele_1 if e not in stop_words_set]
        ele_0 = [e for e in ele_0 if e not in stop_words_set]
        ele_c1 = Counter(ele_1)  # 计频数
        ele_c0 = Counter(ele_0)
        s_1 = sum(ele_c1.values())
        s_0 = sum(ele_c0.values())
        ele_c = {k: (ele_c1[k] / s_1 - ele_c0.get(k, 0) / s_0) for k in ele_c1}  # 计算频率差
        ele_l = [(k, v) for k, v in ele_c.items()]
        ele = sorted(ele_l, key=lambda a: a[1], reverse=True)
        ele = [e[0] for e in ele]
        return ele[:first_n]  # 取前first_n个

    def create_pattern_elements(self, selected_elements, repeat_rate):
        """创建正则表达式元素列表
        :param selected_elements: 可用于作为正则表达式元素的词语列表
        :param repeat_rate: '.*'，''与elements的比例
        :return: None
        """
        repeat_times = int(len(selected_elements) * repeat_rate)
        # self.pattern_elements.extend(['(?:' + x + ')*' for x in selected_elements])
        # self.pattern_elements.extend(['(?:' + x + ')+' for x in selected_elements])
        self.pattern_elements.extend(['(' + x + ')+' for x in selected_elements])
        self.pattern_elements.extend([''] * (repeat_times * 5 + 1))
        self.pattern_elements.extend(['.*'] * (repeat_times + 1))
        self.pattern_elements_len = len(self.pattern_elements)

    def fit(self, strings, elements, labels, stop_words_set):
        """对于给定的strings,elements,labels，获取满足要求的一组正则表达式
        :param strings: 字符串列表
        :param elements: 分词元素列表
        :param labels: label列表
        :param stop_words_set: 停用词集合
        :return: 一组正则表达式
        """
        self.strings = strings
        self.max_str_len = max([len(s) for s in self.strings])
        self.labels = labels
        self.l1_set = set([i for i in range(len(self.labels)) if self.labels[i] == 1])
        self.l0_set = set([i for i in range(len(self.labels)) if self.labels[i] == 0])
        # todo: 如何设置first_n和repeat_rate
        first_n = 10
        repeat_rate = 0.5
        selected_elements = self.get_pattern_elements(elements, stop_words_set, first_n)
        self.create_pattern_elements(selected_elements, repeat_rate)
        gen = 0
        p = self.init()
        best_tpr = 0.0
        best_tpr_fpr = 0.0
        best_pattern_strs = set()
        tprs = []
        while True:
            logging.getLogger('out0').debug('\n\n' + "-" * 100)
            logging.getLogger('out0').info("generation: %d" % gen)
            # p = np.array(list(set([tuple(x) for x in list(p)])))
            es = self.evaluate(p)
            ess = [(i, es[i]) for i in range(len(es))]
            es_sort = sorted(ess, key=lambda a: a[1], reverse=True)
            best_es_this_gen = 0
            for e in es_sort:
                best_es_this_gen = e[1]
                best_pattern_code = p[e[0]]
                best_pattern_str = self.decode(best_pattern_code)
                if best_pattern_str not in best_pattern_strs:
                    break
            for e in es_sort:
                logging.getLogger('out0').debug('%s\t%.2f' % (self.decode(p[e[0]]), e[1]))
            best_pattern_strs.add(best_pattern_str)
            logging.getLogger('out0').debug("generation: %d\nbest_pattern: %s\nbest_es: %f" % (gen, best_pattern_str, best_es_this_gen))
            valid_p_cp = copy.deepcopy(self.valid_patterns)
            self.valid_patterns[best_pattern_str] = self.get_match_results(best_pattern_str)
            tpr, fpr = self.test_valid()
            tprs.append(tpr)
            if tpr > best_tpr and fpr < 0.2:
                best_tpr = tpr
                best_tpr_fpr = fpr
            else:
                self.valid_patterns = valid_p_cp
            logging.getLogger('out0').info('tpr: %f\tfpr: %f' % (best_tpr, best_tpr_fpr))
            logging.getLogger('out0').debug("valid patterns:")
            for pattern, lss in self.valid_patterns.items():
                logging.getLogger('out0').debug('\t%s\t%s' % (pattern, len(lss)))
            if len(tprs) > 10 and tpr >= 0.6 and tpr - tprs[-10] < 0.01:
                return [['.*' + e + '.*' for e in list(self.valid_patterns.keys())], gen, (best_tpr, best_tpr_fpr)]
            if gen > self.max_gen:
                return [['.*' + e + '.*' for e in list(self.valid_patterns.keys())], gen, (best_tpr, best_tpr_fpr)]
            p, es = self.sample_p(p, es)
            parent_indexes = self.parent_select(p)
            children = self.crossover(p, list(parent_indexes))
            p = np.concatenate([p, children])
            mutate_ones = self.mutate(p)
            p = np.concatenate([p, mutate_ones])
            gen += 1
            logging.getLogger('out0').handlers[0].flush()
            logging.getLogger('out0').handlers[1].flush()


def get_data(data_csv_name, stop_words_file_name, label):
    """获取data
    :param data_csv_name: data文件名
    :param stop_words_file_name: 停用词文件名
    :param label: 要匹配的label
    :return: 字符串列表，分词列表，label列表，停用词集合
    """
    strings, elements, labels = [], [], []
    with open(data_csv_name, encoding='utf-8')as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            strings.append(row[0])
            elements.append(row[1].split(' '))
            # 与给定label相同的label设为1，其他设为0
            labels.append(1 if int(row[2]) == label else 0)
    stop_words = set()
    with open(stop_words_file_name, encoding='utf-8')as f:
        for line in f:
            stop_words.add(line.strip())
    stop_words.add('')
    logging.getLogger('out1').debug("info strings\n%s" % strings)
    logging.getLogger('out1').debug("info elements\n%s" % elements)
    logging.getLogger('out1').debug('info labels\n%s' % labels)
    return strings, elements, labels, stop_words


class LevelFilter(logging.Filter):
    """自定义的日志filter，按级别过滤
    """

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno >= self.level


def set_log(t_label):
    log_path = './logs/'
    out0_file_name = log_path + 'label' + str(t_label) + '_out0.log'
    out1_file_name = log_path + 'label' + str(t_label) + '_out1.log'
    try:
        os.remove(out0_file_name)
    except os.error:
        pass
    try:
        os.remove(out1_file_name)
    except os.error:
        pass
    out0_handler = logging.FileHandler(out0_file_name, 'w', encoding='utf-8')
    out0_handler.addFilter(LevelFilter(logging.DEBUG))
    out0 = logging.getLogger('out0')
    out0.addHandler(out0_handler)
    print_handler = logging.StreamHandler()
    print_handler.addFilter(LevelFilter(logging.INFO))
    out0.addHandler(print_handler)
    out0.setLevel(logging.DEBUG)

    out1_handler = logging.FileHandler(out1_file_name, 'w', encoding='utf-8')
    out1 = logging.getLogger('out1')
    out1.addHandler(out1_handler)
    out1.setLevel(logging.DEBUG)
    return out0, out1


def main(t_l):
    target_label = t_l
    set_log(target_label)
    strings, elements, labels, stop_words_set = get_data(
        'data/data_all.csv', 'data/stop_words.txt', target_label)

    population_num = 200
    dim = int(max([len(st) for st in elements]) / 1)
    max_gen = 1000
    crossover_rate = 0.2
    mutate_rate = 0.1
    div = 1
    dtype = np.uint16
    logging.getLogger('out0').info('target_label:' + str(target_label))
    logging.getLogger('out0').info('-------------------------------运行参数如下-------------------------------')
    logging.getLogger('out0').info("population_num:%d\n"
                                   "dim:%d\n"
                                   "max_gen:%d\n"
                                   "crossover_rate:%f\n"
                                   "mutate_rate:%f\n"
                                   "div:%.2f\n\n" % (population_num, dim, max_gen, crossover_rate, mutate_rate, div))
    ere = EvolutionRe(population_num, dim, max_gen, crossover_rate, mutate_rate, div, dtype)
    ps, gen, pr = ere.fit(strings, elements, labels, stop_words_set)
    sav_path = './savs/'
    with open(sav_path + 'label' + str(target_label) + '_sav.json', 'w') as sav_f:
        json.dump({'target_label': target_label, 'ps': ps, 'gen': gen, 'pr': pr}, sav_f, indent=1)
    logging.getLogger('out0').info('\n\n-------------------结果如下--------------------')
    logging.getLogger('out0').info("total generation:\t%d" % gen)
    logging.getLogger('out0').info("valid patterns:")
    for pattern in ps:
        logging.getLogger('out0').info('\t%s' % pattern)
    logging.getLogger('out0').info("tpr: %.2f,fpr: %.2f" % (pr[0], pr[1]))


if __name__ == '__main__':
    t_l_main = int(sys.argv[1])
    main(t_l_main)
