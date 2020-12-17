Q training using keras and tensorflow
====

Overview

- Train invador-like game with Q training
- keras + tensorflow
- 2 patterns of network: full connected and convolution

## Requirement

- Ubuntu 18.04 LTS
- python 3.6.9
- tensorflow 1.14.0
- keras 2.3.1

## Usage

1. start train.py for training

2. start test.py for checking result

Note:
- if you want to try 8 x 8 field, modify train.py as:

```
# environment, agent
env = Invador(simple=True)" 
```

- if you want to try full connected network, modify dqn_agent.py as:

```
self.model_inputs, self.model_outputs, self.model = build_model(input_shape, len(self.enable_actions))        
self.target_model_inputs, self.target_model_outputs, self.target_model = build_model(input_shape, len(self.enable_actions))
```

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)