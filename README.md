#### 简介
在offline数据集上用数据训练一个dynamics model，然后直接拿这个dynamics model当模拟器，跑online RL算法。任意形式的dynamics model组合任意一种online RL算法。
- 当前支持的dynamics model: ADM即any-step dynamics model (后面会加入ensemble dynamics model以前其他baseline)
- 当前支持的online RL算法: SAC, TD3, PPO (PPO目前还有问题，会出现NaN)

#### pipeline
主要是两种:
- 先用数据集训dynamics model，然后用dynamics model训策略
- 加载之前训的dynamics model，然后用dynamics model训策略

#### 主要参数
所有参数都在main.py里头。

##### task相关参数
- ```env```: benchmark，"d4rl"或"neorl"
- ```env-name```: 具体task，例如"hopper-medium-v2"

##### 模型训练相关参数
- ```dyna-model```: dynamics model类型，目前只支持"adm"，就是any-step dynamics model
- ```model-hidden-him```: dynamics model隐层神经元个数
- ```model-lr```: dynamics model学习率
- ```rollout-batch-size```: 使用model进行rollout的batch size，相当于并行模拟器的个数
- ```rollout-length```: rollout长度，这里是要把model直接当模拟器，所以取MuJoCo的默认episode length即1000
- ```max-adm-step```: 只对ADM有效，ADM的最大回溯步长

##### 模型加载相关
如果需要加载已经训练好的模型，可以用这些参数来加载，这样就不会训dynamics model了。一般训练好的dynamics model的pth存放在"result/{env}/{env_name}/{load-label}/{load-time}/model/dyna_seed-{load-seed}.pth"
这里我们需要对这个路径进行指定。
- ```load-model```: 是否加载训练好的dynamics model，True or False
- ```load-label```: 比如之前训练了一版adm + sac组合的，加载的时候load-label就是"adm-sac"
- ```load-time```: 保存模型的时候会用开始训练时的时间戳来进行保存，复制一下文件名的字符串即可
- ```load-seed```: 要加载的种子

这些变量都是训一版有result路径之后自然就知道怎么填了，只在不想从头训练dynamics model，想加载已有model的时候使用。

##### RL相关
- ```algo```: 要使用的RL算法，"sac" or "td3" or "ppo"
- ```ac-hidden-dims```: actor和critic的隐层神经元，比如[256, 256]代表有两个隐层，每个层256个神经元
- ```actor-lr```: actor的学习率
- ```lr-schedule```: 使用使用学习率的schedule，只对actor有效（来自皓哥的offline RL trick）
- ```critic-lr```: critic的学习率
- ```gamma```: discount factor
- ```tau```: target network软更新率
- ```penalty-coef```: model-based rl一般会在reward中加一个不确定性的惩罚项，这个是系数
- 剩下的就是一些RL算法特定的参数，比如SAC的alpha，PPO的GAE lambda

##### 训练相关
- ```buffer-size```: replay buffer大小。注意！这个不是用来存放offline dataset的buffer，而是把model当模拟器，然后策略在model里头采样用于存放的buffer。该参数只对sac和td3有效，ppo有自己专门的on-policy buffer。
- ```warmup-steps```: 一般online RL会先在simulator里面采样，当样本数量达到一定阈值后，才会开始训练，这个表示开始训练的步数，这里默认取10，然后model rollout并行是4096，相当于说4096*10个样本之后才会开始更新策略。
- ```n-epochs```: 训练的总epochs数
- ```step-per-epoch```: 每个epoch的步数，这里取比较短24，因为并行开比较多4096，一个epoch也有将近10w样本了
- ```updates-per-step```: 每一步采样的更新次数，因为一步采样默认对应4096个样本，这里不能只更新一次，暂时是更新20次
- ```batch-size```: 策略更新一次的样本数量
- ```eval-n-episodes```: 每个epoch结束对策略进行评估，评估的episode数量
- ```device```: cpu或cuda
- ```seed```: 这个参数目前是无效的，设置种子请改seeds
- ```seeds```: 比如一次跑5个种子，可以设置成[0, 1, 2, 3, 4]，这里是串行跑的，跑完一个才会跑下一个

考虑到每个task可能使用不同的参数，改参数可以在config目录下的配置文件那里设置。

#### 文件结构

##### agent
agent文件夹下面存放的是RL算法，目前是sac，td3和ppo这三个

##### buffer
buffer文件夹下面存放的是replay buffer，目前也是三个
- ```buffer.py```: 最普通的replay buffer
- ```buffer4rollout```: 这个是给ppo用的，参考tianshou的实现
- ```buffer4seqsamp```: 这个支持多步采样，给adm学习用的

##### components
actor，critic还有基本的mlp网络结构。static_fns也在这里，static_fn指的是每个任务的done函数，都是参考gym官网的done函数。

##### config
配置文件。每个配置文件命名成{env_name}.yml，存放在{env}路径下。比如hopper-medium.yml存放在config/d4rl下面。

##### dynamics
dynamics model在这里。目前只有ADM

##### env
这个指的是学习好的dynamics model，把这个model当模拟器的话需要有一个封装的gym.Env

##### runner
主要就是mas_trainer.py，整个pipeline都在里面实现。ope_runner.py是想后面跑ope的实验。