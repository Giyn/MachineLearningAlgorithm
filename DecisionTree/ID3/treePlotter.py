import matplotlib.pyplot as plt


# 定义文本框和箭头格式,返回相应的字典
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    函数作用：绘制带箭头的注解,createPlot.axl是一个全局变量
    :param node_txt: 需要添加的注解文字
    :param center_pt: 箭头的首部坐标
    :param parent_pt: 箭头的尾部坐标
    :param node_type: 节点的类型(字典的名称)
    :return: None
    """
    create_plot.axl.annotate(node_txt, xy=parent_pt, xycoords="axes fraction",
                             xytext=center_pt, textcoords="axes fraction", va="center",
                             ha="center", bbox=node_type, arrowprops=arrow_args)


"""
:Note: 决策树的叶节点，即没有分支的结点，如下图的B,C
       决策树的叶节点，即决策树的宽度，下图的宽度为2
       A是决策树的判断结点
       决策树的深度，即高度，下图的深度为1
:图例决策时：       A
                 / \
                B   C
"""
def get_num_leafs(my_tree):
    """
    函数作用：获取决策树的叶节点数目
    :param my_tree: 数结构字典
    :return: 树的叶节点数目
    """
    # 叶节点数目
    num_leafs = 0
    # 第一个判断结点
    first_str = list(my_tree.keys())[0]
    # 判断结点的左右分支(也是字典/决策树)
    second_dict = my_tree[first_str]
    # 遍历判断结点的左右分支的keys
    for key in second_dict.keys():
        # key对应的value的类型是字典，则递归调用函数
        # 否则，叶结点数目加1
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    """
    函数作用：获得决策树的深度
    :param my_tree: 给定的决策树
    :return:
    """
    # 初始化深度
    max_depth = 0
    # 第一个判断结点
    first_str = list(my_tree.keys())[0]
    # 判断结点的左右分支(也是字典/决策树)
    second_dict = my_tree[first_str]
    # 遍历判断结点的左右分支的keys
    for key in second_dict.keys():
        # key对应的value的类型是字典，则递归调用函数，并深度加1
        # 否则，深度为1
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrive_tree(i):
    """
    函数作用：定义树结构，测试树结构函数
    :param i: 树列表的index
    :return: 返回list[index]
    """
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers':
                                      {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers':
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                    ]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parant_pt, txt_string):
    """
    函数作用：计算父节点和子结点的中间位置，并在此处添加文本标签信息
    :param cntr_pt: 子结点的坐标
    :param parant_pt: 父结点的坐标
    :param txt_string: 文本标签信息
    :return: None
    """
    # 父节点和子结点的中间位置
    x_mid = (parant_pt[0] - cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parant_pt[1] - cntr_pt[1])/2.0 + cntr_pt[1]
    # 在中间位置处添加文本标签信息
    create_plot.axl.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parant_pt, node_txt):
    """
    函数作用：
    :param my_tree: 决策树
    :param parant_pt: 父节点
    :param node_txt: 结点文本标签信息
    :return:
    """
    # 计算叶节点
    num_leafs = get_num_leafs(my_tree)
    # 计算深度
    depth = get_tree_depth(my_tree)
    # 第一个判断节点
    first_str = list(my_tree.keys())[0]
    # 初始化子节点的位置
    # 全局变量plot_tree.totalW存储树的宽度
    # 全局变量plot_tree.totalD存储树的高度
    cntr_pt = (plot_tree.xoff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW,
               plot_tree.yoff)
    # 计算父节点和子节点的中间位置，并在此处添加文本标签信息
    plot_mid_text(cntr_pt, parant_pt, node_txt)
    plot_node(first_str, cntr_pt, parant_pt, decisionNode)
    # 第一个判断节点的value
    second_dict = my_tree[first_str]
    # 按比例减少全局变量plot_tree.yoff
    plot_tree.yoff = plot_tree.yoff - 1.0/plot_tree.totalD
    # 遍历second_dict.keys()，绘制子节点(可以是叶子结点，或判断节点)
    for key in second_dict.keys():
        # key对应的value的类型是字典(说明是判断节点)，则递归调用函数
        # 否则(说明是叶子节点)，增加X的偏移，绘制箭头，并计算中间位置，添加文本信息
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xoff = plot_tree.xoff + 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xoff, plot_tree.yoff),
                      cntr_pt, leafNode)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntr_pt, str(key))
    # 绘制完所有子节点之后，增加全局变量Y的偏移
    plot_tree.yoff = plot_tree.yoff + 1.0/plot_tree.totalD


def create_plot(in_tree):
    """
    函数作用：创建新图形并清空绘图区，在绘图区绘制决策节点和叶节点
    :param in_tree: 决策树
    :return:
    """
    # 创建窗口
    fig = plt.figure(1, facecolor='white')
    # 清空绘图区
    fig.clf()
    # 创建空字典
    axprops = dict(xticks=[], yticks=[])
    # 创建子图的绘图区
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)
    # 全局变量plot_tree.totalW存储决策树的宽度
    # 全局变量plot_tree.totalD存储决策树的高度
    # 全局变量plot_tree.xoff存储决策树节点的x坐标
    # 全局变量plot_tree.yoff存储决策树节点的y坐标
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xoff = -0.5/plot_tree.totalW
    plot_tree.yoff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')  # 决策树的起点：(0.5,1.0)
    # 调用"绘制带箭头的注解"函数
    # plot_node('decisionNodes', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plot_node('leafNodes', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# if __name__ == '__main__':
#     myTree = retrive_tree(1)
#     create_plot(myTree)
