<h1>Neuroevolutionary Multitasking (NuEMT)</h1>

<h3>Welcome to the implementation for NuEMT algorithm for reinforcement learning</h3>

NuEMT is a new EMT algorithm that builds on the [OpenAI-ES algorithm](https://arxiv.org/pdf/1703.03864.pdf) for continuous control tasks, based on the paper "[Multitask Neuroevolution for Reinforcement Learning with Long and Short Episodes](https://ieeexplore.ieee.org/document/9950429)"

The NuEMT implementation relies on the Ray library (1.2.0) for parallel computing and follows the implementation style in this [repo](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links).

<h4>Note that:</h4>

In our implementation, we find it useful to perform [state normalization](https://proceedings.neurips.cc/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf) 
as it enables different state components to have a fair share of influence during training. A similar normalization approach known as virtual batch normalization is also used by OpenAI-ES. 
In addition, weight decay is added as a form of regularization to prevent parameters of the policy network from exploding. Lastly, we adopt mirror sampling as a variance reduction technique
