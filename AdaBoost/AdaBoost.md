[TOC]

# 集成学习

## 个体与集成

​	集成学习(ensemble learning)通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统(multi-classifer system)、基于委员会的学习(committee-based learning)等。



​	下图显示出集成学习的一般结构：先产生一组“个体学习器”(individual learner)，再用某种策略将它们结合起来。个体学习器通常由一个现有的学习算法从训练数据产生，例如C4.5决策树算法、BP神经网络算法等，此时集成中只包含同种类型的个体学习器，例如“决策树集成”中全是决策树，“神经网络集成”中全是神经网络，这样的集成是“同质的(homogeneous)。同质集成中的个体学习器亦称“基学习器”(base learner)，相应的学习算法称为“基学习算法”(base learning algorithm)。

​	集成也可包含不同类型的个体学习器，例如同时包含决策树和神经网络，这样的集成是“异质”的(heterogenous)。异质集成中的个体学习器由不同的学习算法生成，这时就不再有基学习算法；相应的，个体学习器一般不称为基学习器，常称为“组件学习器”(component learner)或直接称为个体学习器。

![集成学习示意图.png](https://github.com/Giyn/QGSummerTraining/blob/master/Pictures/AdaBoost/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E7%A4%BA%E6%84%8F%E5%9B%BE.png?raw=true)

​	集成学习通过将多个学习器进行结合，常可获得比单一学习器显著优越的泛化性能。这对“弱学习器”(weak learner)尤为明显，因此集成学习的很多理论研究都是针对弱学习器进行的，而基学习器有时也被直接称为弱学习器。但需注意的是，虽然从理论上来说使用弱学习器集成足以获得好的性能，但在实践中出于种种考虑，例如希望使用较少的个体学习器，或是重用关于常见学习器的一些经验等，人们往往会使用比较强的学习器。

>弱学习器常指泛化性能略优于随机猜测的学习器；例如在二分类问题上精度略高于50%的分类器。

​	在一般经验中，如果把好坏不等的东西掺到一起，那么通常结果会是比最坏的要好一些，比最好的要坏一些。集成学习把多个学习器结合起来，如何能获得比最好的单一学习器更好的性能呢？

​	考虑一个简单的例子：在二分类任务中，假定三个分类器在三个测试样本上的表现如下图所示，其中 $√$ 表示分类正确，$×$ 表示分类错误，集成学习的结果通过投票法(voting)产生，即“少数服从多数”。

​	在下图(a)中，每个分类器都只有 $66.6\%$ 的精度，但集成学习却达到了 $100\%$；在下图(b)中，三个分类器没有差别，集成之后性能没有提高；在下图(c)中，每个分类器的精度都只有33.3%，集成学习的结果变得更糟。这个简单的例子显示出：要获得好的集成，个体学习器应“好而不同”，即个体学习器要有一定的“准确性”，即学习器不能太坏，并且要有“多样性”(diversity)，即学习器间具有差异。

>个体学习器至少不差于弱学习器。

![集成个体应“好而不同”(h_i表示第i个分类器).png](https://github.com/Giyn/QGSummerTraining/blob/master/Pictures/AdaBoost/%E9%9B%86%E6%88%90%E4%B8%AA%E4%BD%93%E5%BA%94%E2%80%9C%E5%A5%BD%E8%80%8C%E4%B8%8D%E5%90%8C%E2%80%9D(h_i%E8%A1%A8%E7%A4%BA%E7%AC%ACi%E4%B8%AA%E5%88%86%E7%B1%BB%E5%99%A8).png?raw=true)

​	我们来做个简单的分析。考虑二分类问题 $y∈\{-1,+1\}$ 和真实函数 $f$，假定基分类器的错误率为 $\epsilon$，即对每个基分类器 $h_i$ 有
$$
P(h_i(x)\neq f(x))=\epsilon\,\,.\tag{1}
$$
假设集成通过简单投票法结合 $T$ 个基分类器，若有超过半数的基分类器正确，则集成分类就正确：
>为简化讨论，假设 $T$ 为奇数。

$$
H(x)=sign\left(\sum^T_{i=1}h_i(x)\right)\,\,.\tag{2}
$$

​	假设基分类器的错误率相互独立，则由 $Hoeffding$ 不等式可知，集成的错误率为
$$
P(H(x)\neq f(x))=\sum^{[T/2]}_{k=0}\begin{pmatrix}T\\k\\\end{pmatrix}(1-\epsilon)^k\epsilon^{T-k}\leq exp\left(-\frac{1}{2}T(1-2\epsilon)^2\right)\,\,.\tag{3}
$$
上式显示出，随着集成中个体分类器数目 $T$ 的增大，集成的错误率将指数级下降，最终趋向于零。

​	然而我们必须注意到，上面的分析有一个关键假设：基学习器的误差相互独立。在现实任务中，个体学习器是为解决同一个问题训练出来的，它们显然不可能相互独立！事实上，个体学习器的“准确性”和“多样性”本身就存在冲突。一般地，准确性很高之后，要增加多样性就需牺牲准确性。事实上，**如何产生并结合“好而不同”的个体学习器，恰是集成学习研究的核心**。

​	根据个体学习器的生成方式，目前的集成学习方法大致可分为两大类，即个体学习器间存在强依赖关系、必须串行生成的序列化方法，以及个体学习器间不存在强依赖关系、可同时生成的并行化方法；前者的代表是 $Boosting$，后者的代表是 $Bagging$ 和“ 随机森林”(Random Forest)。



## $Boosting$

​	$Boosting$ 是一族可将弱学习器提升为强学习器的算法。这族算法的工作机制类似：先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值 $T$，最终将这 $T$ 个基学习器进行加权结合。
​	$Boosting$ 族算法最著名的代表是 $AdaBoost$，其描述如下图所示，其中 $y_i∈\{-1,+1\}$，$f$ 是真实函数。

![AdaBoost算法.png](https://github.com/Giyn/QGSummerTraining/blob/master/Pictures/AdaBoost/AdaBoost%E7%AE%97%E6%B3%95.png?raw=true)



### $AdaBoost$

​	$AdaBoost$ 算法有多种推导方式，比较容易理解的是基于“加性模型”(additive model)，即基学习器的线性组合
$$
H(x)=\sum^T_{t=1}α_th_t(x)\tag{4}
$$
来最小化指数损失函数(exponential loss function)
$$
ℓ_{exp}(H\mid D)=\mathbb{E}_{x\sim D}[e^{-f(x)H(x)}]\,\,.\tag{5}
$$

> 符号 $\mathbb{E}_{x\sim D}[.]$ 的含义：
>
> $D$ 为概率分布，可简单理解为在数据集 $D$ 中进行一次随机抽样，每个样本被取到的概率；$\mathbb{E}[.]$ 为经典的期望，则综合起来 $\mathbb{E}_{x\sim D}[.]$ 表示在概率分布 $D$ 上的期望，可简单理解为对数据集 $D$ 以概率 $D$ 进行加权后的期望。

若 $H(x)$ 能令指数损失函数最小化，则考虑式 $(5)$ 对 $H(x)$ 的偏导
$$
\frac{\partial ℓ_{exp}(H\mid D)}{\partial H(x)}=-e^{-H(x)}P(f(x)=1\mid x)+e^{H(x)}P(f(x)=-1\mid x)\,\,,\tag{6}
$$
令式 $(6)$ 为零可解得
$$
H(x)=\frac{1}{2}ln\frac{P(f(x)=1\mid x)}{P(f(x)=-1\mid x)}\,\,,\tag{7}
$$
因此，有
$$
\begin{align}
sign(H(x))&=sign\left(\frac{1}{2}ln\frac{P(f(x)=1\mid x)}{P(f(x)=-1\mid x)}\right)\\
&=\begin{cases}
1,\quad{}&P(f(x)=1\mid x)＞P(f(x)=-1\mid x)\\
-1,\quad{}&P(f(x)=1\mid x)＜P(f(x)=-1\mid x)
\end{cases}\\
&=\arg\limits_{y∈\{-1,1\}}maxP(f(x)=y\mid x)\,\,,\tag{8}
\end{align}
$$

> 此处忽略 $P(f(x)=1\mid x)=P(f(x)=-1\mid x)$ 的情形。

这意味着 $sign (H(x))$ 达到了贝叶斯最优错误率。换言之，若指数损失函数最小化，则分类错误率也将最小化；这说明指数损失函数是分类任务原本 $0/1$ 损失函数的一致的(consistent)替代损失函数。由于这个替代函数有更好的数学性质，例如它是连续可微函数，因此我们用它替代 $0/1$ 损失函数作为优化目标。

​	在 $AdaBoost$ 算法中，第一个基分类器 $h_1$ 是通过直接将基学习算法用于初始数据分布而得；此后迭代地生成 $h_t$ 和 $α_t$，当基分类器 $h_t$ 基于分布 $D_t$ 产生后，该基分类器的权重 $α_t$ 应使得 $α_th_t$ 最小化指数损失函数
$$
\begin{align}
ℓ_{exp}(α_th_t\mid D_t)&=\mathbb{E}_{x\sim D_t}[e^{-f(x)α_th_t(x)}]\\
&=\mathbb{E}_{x\sim D_t}[e^{-α_t}Ⅱ(f(x)=h_t(x))+e^{α_t}Ⅱ(f(x)≠h_t(x))]\\
&=e^{-α_t}P_{x\sim D_t}(f(x)=h_t(x))+e^{α_t}P_{x\sim D_t}(f(x)≠h_t(x))\\
&=e^{-α_t}(1-\epsilon_t)+e^{α_t}\epsilon_t\,\,,\tag{9}
\end{align}
$$
其中 $\epsilon_t=P_{x\sim D_t}(h_t(x)≠f(x))$。考虑指数损失函数的导数
$$
\frac{\partial ℓ_{exp}(α_th_t\mid D_t)}{\partial α_t}=-e^{-α_t}(1-\epsilon_t)+e^{α_t}\epsilon_t\,\,,\tag{10}
$$
令式 $(10)$ 为零可解得
$$
α_t=\frac{1}{2}ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)\,\,,\tag{11}
$$
这恰是上图算法第 $6$ 行的分类器权重更新公式。

​	$AdaBoost$ 算法在获得 $H_{t-1}$ 之后样本分布将进行调整，使下一轮的基学习器 $h_t$ 能纠正 $H_{t-1}$ 的一些错误。理想的 $h_t$ 能纠正 $H_{t-1}$ 的全部错误，即最小化
$$
\begin{align}
ℓ_{exp}(H_{t-1}+h_t\mid D)&=\mathbb{E}_{x\sim D}[e^{-f(x)(H_{t-1}(x)+h_t(x))}]\\
&=\mathbb{E}_{x\sim D}[e^{-f(x)H_{t-1}(x)}e^{-f(x)h_t(x)}]\,\,.\\\
\end{align}\tag{12}
$$
注意到 $f^2(x)=h^2_t(x)=1$，式 $(12)$ 可使用 $e^{-f(x)h_t(x)}$ 的泰勒展式近似为

> $e^x$ 的泰勒展开公式：$e^x=1+\frac{1}{1!}x+\frac{1}{2!}x^2+\frac{1}{3!}x^3+o(x^3)$。

$$
\begin{align}
ℓ_{exp}(H_{t-1}+h_t\mid D)&≈\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\left(1-f(x)h_t(x)+\frac{f^2(x)h^2_t(x)}{2}\right)\right]\\
&=\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\left(1-f(x)h_t(x)+\frac{1}{2}\right)\right]\,\,.\tag{13}
\end{align}
$$

于是，理想的基学习器
$$
\begin{align}
h_t(x)&=\arg\limits_hmin\,ℓ_{exp}(H_{t-1}+h\mid D)\\
&=\arg\limits_hmin\,\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\left(1-f(x)h_t(x)+\frac{1}{2}\right)\right]\\
&=\arg\limits_hmax\,\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}f(x)h(x)\right]\\
&=\arg\limits_hmax\,\mathbb{E}_{x\sim D}\left[\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\right]}f(x)h(x)\right]\,\,,\tag{14}
\end{align}\
$$
注意到 $\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\right]$ 是一个常数。令 $D_t$ 表示一个分布
$$
D_t(x)=\frac{D(x)e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\right]}\,\,,\tag{15}
$$
则根据数学期望的定义，这等价于令
$$
\begin{align}
h_t(x)&=\arg\limits_hmax\,\mathbb{E}_{x\sim D}\left[\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\right]}f(x)h(x)\right]\\
&=\arg\limits_hmax\,\mathbb{E}_{x\sim D}[f(x)h(x)]\,\,.\tag{16}
\end{align}
$$
由 $f(x),h(x)∈\{-1,+1\}$，有
$$
f(x)h(x)=1-2Ⅱ(f(x)\neq h(x))\,\,,\tag{17}
$$
则理想的基学习器
$$
h_t(x)=\arg\limits_hmin\,\mathbb{E}_{x\sim D}[Ⅱ(f(x)\neq h(x))]\,\,.\tag{18}
$$
由此可见，理想的 $h_t$ 将在分布 $D_t$ 下最小化分类误差。因此，弱分类器将基于分布 $D_t$ 来训练，且针对 $D_t$ 的分类误差应小于 $0.5$。这在一定程度上类似**“残差逼近”**的思想。考虑到 $D_t$ 和 $D_{t+1}$ 的关系，有
$$
\begin{align}
D_{t+1}(x)&=\frac{D(x)e^{-f(x)H_{t}(x)}}{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t}(x)}\right]}\\
&=\frac{D(x)e^{-f(x)H_{t-1}(x)}e^{-f(x)α_th_t(x)}}{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t}(x)}\right]}\\
&=D_t(x)\,·\,e^{-f(x)α_th_t(x)}\frac{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t-1}(x)}\right]}{\mathbb{E}_{x\sim D}\left[e^{-f(x)H_{t}(x)}\right]}\,\,,\tag{19}
\end{align}
$$
这恰是上图中算法第 $7$ 行的样本分布更新公式。

​	于是，由式 $(11)$ 和 $(19)$ 可见，我们**从基于加性模型迭代式优化指数损失函数的角度推导出了 $AdaBoost$ 算法**。

​	$Boosting$ 算法对基学习器的要求：

- 对特定的数据分布进行学习，这可通过“重赋权法”(re-weighting)实 施，即在训练过程的每一轮中，根据样本分布为每个训练样本重新赋予一个权重。
- 对无法接受带权样本的基学习算法，则可通过“重采样法”(re- sampling)来处理，即在每一轮学习中，根据样本分布对训练集重新进行采样，再用重采样而得的样本集对基学习器进行训练。

​	一般而言，上述两种做法没有显著的优劣差别。

​	需注意的是，$Boosting$ 算法在训练的每一轮都要检查当前生成的基学习器是否满足基本条件（例如上图算法的第 $5$ 行，检查当前基分类器是否是比随机猜测好），一旦条件不满足，则当前基学习器即被抛弃，且学习过程停止。在此种情形下，初始设置的学习轮数 $T$ 也许还远未达到，可能导致最终集成中只包含很少的基学习器而性能不佳。若采用“重采样法”，则可获得“重启动”机会以避免训练过程过早停止，即在抛弃不满足条件的当前基学习器之后，可根据当前分布重新对训练样本进行采样，再基于新的采样结果重新训练出基学习器，从而使得学习过程可以持续到预设的 $T$ 轮完成。

​	从偏差—方差分解的角度看，$Boosting$ 主要关注降低偏差，因此 $Boosting$ 能基于泛化性能相当弱的学习器构建出很强的集成。

