(1)数据集：
使用的数据集分别是openoffice，eclipse，netBeans，压缩过后的版本以.rar的格式存储，下载解压即可；
deal_datasetexample.ipynb展示了处理数据集的流程示例，component.ipynb展示了某些属性分析流程；

（2）DBRD.py展示了使用的网络模型结构，embedding.ipynb和tokenize.ipynb展示了如何通过bert预处理模型获得对应的嵌入向量，
normalization.ipynb展示了对惩罚项的处理流程；

（3）bert-mlp.py 和 dc-cnn.py中可以找到对比方法的模型结构，dataprepara-dccnn.py 和 test.ipynb分别展示了如何将dataset处理后
输入到dc-cnn模型和bert-mlp模型
