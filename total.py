# encoding=utf-8
import json
import csv
import re


def get_data(data_csv_name,):
    """获取data
    :param data_csv_name: data文件名
    :return: 字符串列表，分词列表，label列表
    """
    strings, elements, labels = [], [], []
    with open(data_csv_name)as csv_f:
        reader = csv.reader(csv_f, delimiter=',')
        next(reader)
        for row in reader:
            strings.append(row[0])
            elements.append(row[1].split(' '))
            labels.append(int(row[2]))
    return strings, elements, labels


def get_pre_label(st, sav_d):
    """获取对一个字符串的预测label
    :param st: 字符串
    :param sav_d: save_dict
    :return: 预测的label
    """
    r = {}
    for label, dic in sav_d.items():
        for pattern in dic['ps']:
            if re.match(pattern, st):
                r[label] = r.get(label, 0) + 1
    if not r:
        return 0
    return max(r, key=r.get)


def write_readme(sav_dict, acc):
    with open('README.md', 'w')as f:
        f.write('\ntotal accuracy: %.2f\n\n' % acc)
        for k, v in sav_dict.items():
            f.write('\n- Label ' + str(k) + ' 的匹配结果\n\n')
            f.write('```python\n')
            f.write('total generation: ' + str(v['gen']) + '\n')
            f.write('valid patterns:\n')
            for p in v['ps']:
                f.write('\t' + p + '\n')
            f.write("tpr: %.2f, fpr: %.2f\n" % (v['pr'][0], v['pr'][1]))
            f.write('```\n')


if __name__ == '__main__':
    sav_path = './savs/'
    sav_dict = {}
    for target_label in [1, 3, 4]:
        with open(sav_path + 'label' + str(target_label) + '_sav.json')as f:
            j = json.load(f)
        sav_dict[target_label] = j
    print(json.dumps(sav_dict, indent=1, ensure_ascii=False))

    strs, eles, labs = get_data('data/data_all.csv')

    pre_labels = []
    for s in strs:
        pre_labels.append(get_pre_label(s, sav_dict))
    a = [1 for i in range(len(labs)) if pre_labels[i] == labs[i]]
    acc = sum(a) / len(labs)
    print('\n\naccuracy: %.2f' % (acc))
    write_readme(sav_dict, acc)
