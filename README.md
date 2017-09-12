# PyTorch-ActorCriticRL

PyTorch implementation of continuous action actor-critic algorithm. The algorithm uses DeepMind's Deep Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971) method for updating the actor and critic networks along with [Ornstein–Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) process for exploring in continuous action space while using a Deterministic policy.

## DDPG

[DDPG](https://arxiv.org/abs/1509.02971) is a policy gradient alogrithm, that uses stochastic behaviour policy for exploration (Ornstein-Uhlenbeck in this case) and outputs a deterministic target policy, which is easier to learn.

### Policy Estimation (Actor)

Actor Network consists of a 3-layer neural network taking into input the state (s) and outputs the action (a) which should be taken denoted by *Pi(s)*.

### Policy Evaluation (Critic)

Critic Network consists of a 3-layer neural network taking into input the state (s) and correspoding action (a) and outputs the state-action value function denoted by __*Q(s,a)*__.

### Actor Optimization

The policy is optimized by minimizing the loss :-  __*sum ( -Q(s,a) )*__.

### Critic Optimization

The critic is optimized by minimzing the loss :- __*L2( r + gamma\*Q(s1,Pi(s)) - Q(s,a) )*__.

### Soft Updates

The above updates however don't tend to converge according to DeepMind's paper and they hence use soft policy updates by maintaing a target actor and critic whose weights are updated after above optimizations as follows :-

```
target_actor = beta*actor + (1-beta)*target_actor
target_critic = beta*critic + (1-beta)*target_critic
```

where beta = 0.001

## Performance of DDPG on OpenAI Envs

### Pendulum-v0

Below is the performance of the model after 70 episodes. [Full Video](https://www.youtube.com/watch?v=feXeEG_KaYw)

![Pendulum-v0](https://j.gifs.com/O71nqL.gif)

### BiPedalWalker-v2

Below is the performance of the model after 900 episodes. [Full Video](https://www.youtube.com/watch?v=-QU42vpBWIg)

![BiPedalWalker-v2](https://j.gifs.com/r0Qx6k.gif)

## References

* [DDPG paper by DeepMind](https://arxiv.org/abs/1509.02971) - DeepMind's DDPG paper
* [DDPG blog by penami4911](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) - A very nicely explained blog with code in Tensorflow 
* [A3C paper by DeepMind](https://arxiv.org/abs/1602.01783) - DeepMind's A3C paper
* [A3C blog by yanpanlau](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html) - A3C explained nicely along with  Ornstein-Uhlenbeck expoloration

