# Dynamic Bottleneck 

## Introduction

This is a TensorFlow based implementation for our paper on 

**"Dynamic Bottleneck for Robust Self-Supervised Exploration". NeurIPS 2021**

## Prerequisites

python3.6 or 3.7,
tensorflow-gpu 1.x, tensorflow-probability,
openAI [baselines](https://github.com/openai/baselines),
openAI [Gym](http://gym.openai.com/)

## Installation and Usage

### Atari games

The following command should train a pure exploration 
agent on "Breakout" with default experiment parameters.

```
python run.py --env BreakoutNoFrameskip-v4
```


### Atari games with Random-Box noise 

The following command should train a pure exploration 
agent on "Breakout" with randomBox noise.

```
python run.py --env BreakoutNoFrameskip-v4 --randomBoxNoise
```

### Atari games with Gaussian noise 

The following command should train a pure exploration 
agent on "Breakout" with Gaussian noise.

```
python run.py --env BreakoutNoFrameskip-v4 --pixelNoise
```


### Atari games with sticky actions

The following command should train a pure exploration 
agent on "sticky Breakout" with a probability of 0.25

```
python run.py --env BreakoutNoFrameskip-v4 --stickyAtari
```

### Baselines

- **ICM**: We use the official [code](https://github.com/openai/large-scale-curiosity) of "Curiosity-driven Exploration by Self-supervised Prediction, ICML 2017" and "Large-Scale Study of Curiosity-Driven Learning, ICLR 2019".   
- **Disagreement**: We use the official [code](https://github.com/pathak22/exploration-by-disagreement) of "Self-Supervised Exploration via Disagreement, ICML 2019".
- **CB**: We use the official [code](https://github.com/whyjay/curiosity-bottleneck) of "Curiosity-Bottleneck: Exploration by Distilling Task-Specific Novelty, ICML 2019".
