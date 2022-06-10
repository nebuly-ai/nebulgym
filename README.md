<img  src="https://user-images.githubusercontent.com/83510798/172371780-484c7efc-ca9d-4881-b4d4-dfee2ae77289.png">

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Contribute to the library</a>
</p>

<p align="center">
<a href="#how-nebulgym-works">How nebulgym works</a> •
<a href="#benchmarks">Benchmarks</a> •
<a href="#installation">Installation</a> •
<a href="#get-started">Get started</a> •
<a href="#nebulgym-use-case">Tutorials & examples</a> •
<a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Documentation</a>
</p>

<p align="center">
<a href="https://nebuly.ai/">Website</a> |
<a href="https://www.linkedin.com/company/72460022/">LinkedIn</a> |
<a href="https://twitter.com/nebuly_ai">Twitter</a>
</p>

# Nebulgym

Easy-to-use library to accelerate AI training leveraging state-of-the-art optimization techniques 🤸‍♀️

-  [How nebulgym works](#how-nebulgym-works)
-  [Benchmarks](#benchmarks)
-  [Tutorials and examples](#nebulgym-use-case)
-  [Installation & get started](#installation)
-  <a href="https://nebuly.gitbook.io/nebuly/nebulgym/how-nebulgym-works">Documentation</a>
-  <a href="https://discord.gg/jyjtZTPyHS">Join the community for AI acceleration</a>

## How nebulgym works

> `nebulgym` greatly reduces the training time of AI models without requiring any modification to the training setup. `nebulgym` optimizes the full training computing stack, from efficient data loading, to faster forward and backward passes, to earlier convergence.

No matter what model, framework, or training recipe you use, with `nebulgym` you speed up training by simply adding nebulgym class decorators to your code. Decorators will make sure that you use your hardware's computing power to the fullest and achieve the shortest possible training time.

<p align="center">
Your code + @nebulgym_class_decorators = superfast training 🏃‍♀️
</p>


### So why nebulgym?

🚀 **Superfast**. The library speeds up training and thus the time to get to production, reduces computing costs and energy consumption.

☘️ **Easy-to-use**. Just add `nebulgym` class decorators to your code and continue programming on your favorite training framework. nebulgym will let you achieve awesome training times.

💥 **Training setup agnostic**. `nebulgym` can be coupled with any model, trainer, or other training technique to achieve a compound effect on training performance.

🦾 **Framework agnostic**. The library aims to support all frameworks (PyTorch, TensorFlow, Hugging Face, Jax, etc.) to allow any developer to continue working with the configuration they are used to. Currently, nebulgym supports PyTorch and we are working on expanding nebulgym capabilities.

💻 **Deep learning model agnostic**. `nebulgym` supports all the most popular architectures such as transformers, LSTMs, CNNs and FCNs.

🤖 **Hardware agnostic**. The library aims to support any artificial intelligence hardware on the market, from general-purpose GPUs and CPUs to hardware accelerators such as FPGAs and ASICs. Currently, nebulgym has been tested on many CPUs and GPUs.

Do you like the library? Leave a ⭐ if you enjoy the project and join [the community](https://discord.gg/RbeQMu886J) where we chat about `nebulgym` and AI acceleration.

And happy training 🏋️

<img src="https://user-images.githubusercontent.com/83510798/172376401-88930367-5d1d-41c8-a617-57c628d09fdc.png">


## Benchmarks
`nebulgym` has just been launched and has been tested on limited use cases. Early results are remarkably good, and it is expected that `nebulgym` will further reduce training time in future releases. At the same time, it is expected that nebulgym may fail in untested cases and provide different results, perhaps greater or worse than those shown below.

We tested nebulgym on the custom model that you can find in the <a href="#nebulgym-use-case">example section</a>. The test consists in a training over 10 epochs and a batch size of 8.

Below are the training times in seconds before `nebulgym` optimization and after its acceleration, and the speedup, which is calculated as the response time of the unoptimized model divided by the response time of the accelerated model.

### Training time in seconds
| **Hardware** | **Not-optimized** | **Accelerated** | **Speedup** |
|---|:---:|:---:|:---:|
| **M1 Pro** | 632.05 | 347.52 | 1.8x |
| **Intel Xeon** | 788.05 | 381.01 | 2.1x |
| **AMD EPYC** | 1547.35 | 1034.37 | 1.5x |
| **NVIDIA T4** | 258.88 | 127.32 | 2.0x |
| __________________ | _________________ | _________________ | _________________ |

Hardware setup
- M1 Pro: Apple M1 Pro 16GB of RAM
- Intel Xeon: EC2 Instance on AWS - t2.large
- AMD EPYC: EC2 Instance on AWS - t4a.large
- NVIDIA T4: EC2 instance on AWS - g4dn.xlarge

How does `nebulgym` perform on your training setup? What do you think about `nebulgym` and what are ways to make it even better? Share your ideas and results with us in the [community chat](https://discord.gg/RbeQMu886J).


## Installation

Installing and using nebulgym is super easy! You can either
- install `nebulgym` from PyPI (with `pip`) or
- install `nebulgym` from source code.

We strongly recommend that you install `nebulgym` in a new environment. You can create and manage your environment using Conda or another virtual environment management application. We tested our installation with `venv` by `conda`.


### Installation from PyPI

```
pip install nebulgym
```

### Source code installation

Clone nebulgym repository to your local machine.

```
git clone https://github.com/nebuly-ai/nebulgym.git
```

Go into the repo and run the setup.py file.
```
cd nebulgym && python setup.py install
```

## Get started

`nebulgym` accelerates training by means of class decorators. A class decorator is a very elegant and non-intrusive method that allows nebulgym to tag your model (`@accelerate_model`) and your dataset (`@accelerate_dataset`) and add functionalities to their classes. When you run a training session, `nebulgym` will greatly reduce the training time of your decorated model. As simple as that!

You can find more information about `nebulgym` class decorators, the parameters they can take as input, and other `nebulgym` classes that can be used as an alternative to decorators in the [documentation](https://nebuly.gitbook.io/nebuly/nebulgym/get-started/advanced-options).

### How to use nebulgym class decorators

Put `nebulgym` class decorators right before defining your dataset and model classes.

- `@accelerate_dataset` must be entered before the dataset definition. nebulgym will cache dataset samples in memory, so that reading these samples after the first time becomes much faster. Caching the dataset makes data loading faster and more efficient, solving what could become the main bottleneck of the whole training process.
- `@accelerate_model` must be entered before the model definition. nebulgym will accelerate both forward and backward propagations by reducing the number of computationally expensive propagation steps and making computations more efficient.

<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/83510798/172379283-b18f75cb-1b85-4f70-91a6-d3973c593cff.gif">
</p>


## nebulgym use case

Here we show an example of how you can easily use `nebulgym` class decorators. To achieve awesome training speed, you can simply add `nebulgym` decorators (`@accelerate_model` and `@accelerate_dataset`) before defining your AI model and dataset.

```
from typing import List, Callable
import torch
from torch.utils.data import Dataset

from nebulgym.decorators.torch_decorators import accelerate_model, accelerate_dataset

# Add nebulgym annotation before defining your model. 
#  This model takes as input an image of resolution 224x224
@accelerate_model()
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._avg_pool = torch.nn.AvgPool2d(4)
        self._linear = torch.nn.Linear(3136, 1024)
        self._relu = torch.nn.ReLU()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self._avg_pool(x).mean(dim=-3).view(-1, 3136)
        x = self._relu(self._linear(x))
        return self._linears(x)


# Add nebulgym annotation before defining your dataset.
@accelerate_dataset()
class CustomDataset(Dataset):
    def __init__(self, img_paths: List[str], labelling_func: Callable, reading_func: Callable):
        self._images = img_paths
        self._labelling_func = labelling_func
        self._reading_func = reading_func

    def __getitem__(self, item):
        img_path = self._images[item]
        label = self._labelling_func(img_path)
        input_tensor = self._reading_func(img_path)
        return input_tensor, label

    def __len__(self):
        return len(self._images)    
```

And that's it. Now, as soon as you perform a training run, `nebulgym` will optimize the full training computing stack, from efficient data loading, to faster forward and backward passes, to earlier convergence. 

## Supported tech & roadmap
`nebulgym` has just been launched, and it is already capable of cutting training time in half. At the same time, it is expected that `nebulgym` may crash or fail in untested use cases. Moreover, the project is in its early stages and there is a lot of room for improvement for `nebulgym` to become a new paradigm for artificial intelligence training.

`nebulgym` aims to support every framework, every model, every hardware, and make the most of your hardware and software capabilities to train your model in a fraction of the time required now. In addition, `nebulgym` will always be extremely easy to use to empower any developer to build powerful AI applications.

`nebulgym` already embeds many great technologies. Below you can find a list of the features already implemented and those that will be implemented soon. More specific tasks can be found in the [issue page](https://github.com/nebuly-ai/nebulgym). 

Any ideas about what could be implemented next? Would you like to contribute to this fantastic library? We welcome any ideas, questions, issues and pull requests! For more info go to the [Documentation](https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions).

### Supported frameworks
- PyTorch
- TensorFlow ([open issue](https://github.com/nebuly-ai/nebulgym/issues/5))

### Supported backends
- **PyTorch**. Default compiler for models trained in PyTorch.
- **Rammer**. Compiler that can be used on Nvidia GPUs.
- ONNX Runtime. Training API leveraging on some techniques developed for inference optimization. It currently supports only Nvidia GPUs.

### Optimization techniques for data loading
- **Cached datasets**. Data loading is for some use cases slow and could become a major bottleneck of the whole training process. nebulgym provides cached dataset to speedup this process by caching data samples in memory, so that reading these samples after the first time becomes much faster.

### Model Optimization techniques
- **Sparsified Back Propagation**. As traditional neural network consumes a significant amount of computing resources during back propagation, leveraging a simple yet effective technique to alleviate this problem. In this technique, only a small subset of the full gradients are computed to update the model parameters and the model still achieves the same effect as the traditional CNN, or even better.
- **Selective-Backprop**. ([open issue](https://github.com/nebuly-ai/nebulgym/issues/11))
- **Layer Replacement** ([open issue](https://github.com/nebuly-ai/nebulgym/issues/10))
- **ModelReshaper** ([open issue](https://github.com/nebuly-ai/nebulgym/issues/9))
- **Distributed training** ([open issue](https://github.com/nebuly-ai/nebulgym/issues/4))
- **Forward gradients** ([open issue](https://github.com/nebuly-ai/nebulgym/issues/3))

### Library installation methods
- From PyPI
- Source code

### Backend installation methods
- From the backend source code
- Automatic installation with an auto-installer ([open issue](https://github.com/nebuly-ai/nebulgym/issues/8))


## Licence
This project is released under the [Apache 2.0 licence](https://github.com/nebuly-ai/nebulgym/blob/main/LICENSE).

---

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Contribute to the library</a>
</p>

<p align="center">
<a href="#how-nebulgym-works">How nebulgym works</a> •
<a href="#benchmarks">Benchmarks</a> •
<a href="#installation">Installation</a> •
<a href="#get-started">Get started</a> •
<a href="#nebulgym-use-case">Tutorials & examples</a> •
<a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Documentation</a>
</p>

<p align="center">
<a href="https://nebuly.ai/">Website</a> |
<a href="https://www.linkedin.com/company/72460022/">LinkedIn</a> |
<a href="https://twitter.com/nebuly_ai">Twitter</a>
</p>
