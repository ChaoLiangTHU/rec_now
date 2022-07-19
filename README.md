# 项目简介
recommend now (rec_now) 是一个推荐算法的基础库，目的在于简化推荐模型的开发。

本项目基于tensorflow2和python3实现，兼容使用tensorflow1.x API的[无量训练框架](https://git.woa.com/deep_learning_framework/NumerousTensorFlow)，并在生产环境中得到了充分验证。
其中，基于in-batch方式计算pairwise的方式，已经在QQ浏览器信息流的精排、粗排、召回排序等环节得到了广泛应用，并使得在线GAUC指标提升1%以上。

## 功能概览
* pairwise loss和listwise loss: 根据分组ID（如用户ID），提取一个batch内的pair（或list），计算pairwise loss（或listwise loss），从而提升GAUC指标。
* 经典推荐算法：如[FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)，[DCN](https://arxiv.org/abs/1708.05123)，[DCN_mix](https://arxiv.org/abs/2008.13535v2)，[CIN](https://arxiv.org/abs/1803.05170)，[CAN](https://arxiv.org/abs/2011.05625)，[SENET](https://arxiv.org/abs/1905.09433)，[STAR](https://arxiv.org/abs/2101.11427)，[MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007)，[PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)等。
* 原创网络结构：如StackedDenseLayer。


## 项目结构
```
rec_now
├─README.md  说明文件
├─rec_now
│  ├─layers 基于keras.layers.Layer的派生层，用于实现常见的推荐算法
|  |  ├─ can_layer.py 实现co-action network
|  |  ├─ cin_layer.py 实现Compressed Interaction Network，并进行了加速
|  |  ├─ dcn_layer.py 实现Deep & Cross Network
|  |  ├─ dcn_mix_layer.py 实现Deep & Cross Network V2
|  |  ├─ fm_layer.py 实现Factorization Machines
|  |  ├─ inner_pnn_layer.py 实现Inner Product-based Neural Networks
|  |  ├─ mmoe_layer.py 实现Multi-gate Mixture-of-Experts，并进行了加速
|  |  ├─ multi_dense_layer.py 加速计算多个参数量相同的Dense层
|  |  ├─ ple_layer.py 实现Progressive Layered Extraction，并进行了加速
|  |  ├─ pooling_layer.py 对常用pooling函数的封装，如mean，sum，min，max等
|  |  ├─ senet_layer.py 实现Squeeze-Excitation network，并新增支持不等长的embeddings
|  |  ├─ stacked_dense_layer.py 个性化Dense层
|  |  └─ star_dense_layer.py 实现STAR Topology Fully-Connected Network
|  |
│  ├─rec_block 推荐系统的基础函数，如attention机制, pairwise loss，listwise loss等
|  |  ├─ attention.py 基于点积、DNN的注意力模型
|  |  ├─ embedding_wise_weight.py 不等长向量级别的权重映射
|  |  ├─ listwise_loss_from_batch.py 基于in-batch的方式生成pairwise loss
|  |  └─ pairwise_loss_from_batch.py 基于in-batch的方式生成listwise loss
|  |
│  └─util 辅助函数
|     ├─ numpy_tools.py numpy的辅助函数
|     └─ tfprint.py 对tf.Print的封装，方便打印tensor的形状，最大最小值等信息
|
└─tests 测试代码，组织结构和rec_now文件夹相同
```


## 约定
* 对于运算过程中的张量，通过行末的注释给出张量的形状，格式为 `# (dim1, dim2, dim3, ...)`
* 注释中张量的形状一般采用简写（比如用B表示batch_size），并在class或函数头部进行说明


## 相关文章
* [AI与推荐技术在腾讯QQ浏览器的应用（排序模块部分）, AICon 全球人工智能与机器学习技术大会（2021）](https://mp.weixin.qq.com/s/EnMT4xVY3LRCyVqqNqr5WQ)

## 谁在使用
* QQ浏览器信息流
  - 精排pointwise, pairwise联合模型
  - 粗排distrill精排模型

