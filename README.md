- EMG进行手势分类。
- 实验测试环境：
  - Ubuntu 16.04
  - NVIDIA GeForce 3090
  - CUDA Version: 11.7
  - Anaconda 4.12.0
  - Python 3.8.13
- 代码运行所需环境详见`requirements.txt`文件。
- 项目文件包括：
  - `main.py`实验提供的主框架，只对rest label做改动，实验中rest类别设置为10而不是0（对训练与验证无影响）；
  - `train_Emg.py`为主要的数据处理、训练、测试代码；
  - `model.py`为实验所用的模型。
- 运行方式：`python main.py`
  - 对于场景一，单个Session的运行时间大概为：6min左右。
  - 对于场景二，单个Session的运行时间大概为：11min左右。