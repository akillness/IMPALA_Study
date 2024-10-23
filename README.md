# IMPALA_Cartpole
RL 과제 구현 소스관리

# RL과제

Implement IMPALA for CartPole-v1 of gym (or gymnasium) with the following requirements.

- Use 'IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures' for training. - Do not use any baseline code for this task.
- It is not necessary to use multi-processing for this task.(You may use just a single process to implement the algorithm.)
- Write a simple report with graphs of the training:
  -  score: sum of rewards in the episode,
  -  critic loss: loss of the value function,
  -  entropy: entropy of the target policy,
  -  min/max/avg importance sampling ratio: min/max/avg value of importance sampling ratio in a batch,
  -  learner batching time: time consumed for making a batch for an update when enough data are given,
  -  learner forward time: forward time consumed for a batch in learner,
  -  learner backward time: backward time consumed for a batch in learner,
  -  and any other values that are helpful to analyze the throughput and bottle neck of the system.


- (Optional) If you implement a distributed learning system as follows, you will have extra points: 
  1) One learner and 4 actors run on different processes. 
  2) The actor doesn't stop collecting data while the learner is updating parameters.

- (Optional) If you tried to maximize the throughput of the system (number of data consumed by the system per second), please explain your method why it is beneficial for a higher throughput. You will have extra points for this.
  
  
- Submit your code and report together in the form of '.zip' file.
- Your code and report is going to be used for the technical interview.
- 제출기한 : October 23, 2024 로부터 **10 일** 후 자정까지 제출 



## 논문리뷰 ( 키워드 정리 )
Actor-Learner -> 폴리시의 gradient를 보내면던 A3C와 다르게 State를 전달함으로서, Policy-lag 에 효과적
Optimisation -> 일반적인 RL은 Conv-net 특징점 / LSTM , Time dimensiont 을 Batch dimension
V-Trace 
 -> 로컬 폴리시 뮤 를  진행중인 폴리시 파이로 얻기 위한 Target 폴리시로 trajectories 를 얻음 ( called behavior policy )
 -> off-policy는 behavior policy로 얻어진 trajectories를 이용하여 다른 폴리시 파이의 Value function을 학습 ( 뮤와 는 다른 ) ( called target policy )


V-trace 알고리즘이 on-policy n-steps Bellman 업데이트로 축소되는 이유는 다음과 같습니다:

### V-trace 알고리즘
V-trace는 off-policy 데이터를 사용하여 정책을 업데이트할 때 발생하는 편향을 줄이기 위해 설계된 알고리즘입니다. 이 알고리즘은 중요도 샘플링(importance sampling)을 사용하여 off-policy 데이터를 on-policy 데이터처럼 사용할 수 있게 합니다.
### On-policy n-steps Bellman 업데이트
On-policy n-steps Bellman 업데이트는 에이전트가 현재 정책을 사용하여 n 단계 동안 환경과 상호작용한 후, 그 결과를 바탕으로 가치 함수를 업데이트하는 방법입니다. 이 방법은 정책이 변경되지 않는다는 가정 하에 작동합니다.
### V-trace가 on-policy n-steps Bellman 업데이트로 축소되는 이유
V-trace 알고리즘은 중요도 샘플링 비율(importance sampling ratio)
ρt =μ(at ∣st )/π(at ∣st )
를 사용하여 off-policy 데이터를 보정합니다. 여기서 π 는 목표 정책(target policy), μ는 행동 정책(behavior policy)입니다. 만약 목표 정책과 행동 정책이 동일하다면, 즉 π=μ라면, 중요도 샘플링 비율 ρt 는 1이 됩니다. 
이 경우, V-trace 알고리즘은 다음과 같이 단순화됩니다:
- V(xt )=rt +γV(xt+1 )


[이 식은 on-policy n-steps Bellman 업데이트와 동일합니다1](https://ar5iv.labs.arxiv.org/html/1802.01561)[2](https://link.springer.com/article/10.1007/s10489-024-05508-9)~. 

따라서, V-trace 알고리즘은 목표 정책과 행동 정책이 동일할 때 on-policy n-steps Bellman 업데이트로 축소됩니다.

이러한 특성 덕분에 V-trace 알고리즘은 off-policy와 on-policy 데이터를 모두 효과적으로 사용할 수 있음


## 개념정리
IMPALA (Importance Weighted Actor-Learner Architectures)는 DeepMind에서 개발한 분산 심층 강화 학습 프레임워크입니다. 이 아키텍처는 대규모 데이터와 긴 학습 시간을 효율적으로 처리하기 위해 설계되었습니다. 다음은 IMPALA의 주요 개념과 구조에 대한 설명

1.  **Actor-Learner 구조**
IMPALA는 여러 개의 Actor와 Learner로 구성됩니다.
* **Actors**: 환경에서 행동을 수행하고 경험을 수집합니다. 각 Actor는 정책을 사용하여 환경과 상호작용하고, 상태, 행동, 보상 등의 데이터를 수집합니다.
* **Learner**: Actor들이 수집한 데이터를 사용하여 정책과 가치 함수를 업데이트합니다. Learner는 중앙 집중식으로 작동하며, 여러 Actor로부터 데이터를 받아 병렬로 학습을 수행합니다.

1. **V-trace Off-Policy Correction**
IMPALA는 V-trace라는 오프-폴리시 보정 방법을 사용합니다. 이는 Actor들이 Learner의 최신 정책과 다소 다른 정책을 사용할 수 있도록 허용하면서도 안정적인 학습을 가능하게 합니다. ~[V-trace는 Actor들이 수집한 데이터의 중요도를 조정하여 학습의 편향을 줄입니다1](https://arxiv.org/abs/1802.01561)~.

1. **효율적인 자원 사용**
IMPALA는 단일 머신에서의 자원 사용을 최적화할 뿐만 아니라, 수천 대의 머신으로 확장할 수 있습니다. ~[이는 데이터 효율성을 유지하면서도 높은 처리량을 달성할 수 있게 합니다](https://arxiv.org/abs/1802.01561)[2](https://paperswithcode.com/paper/impala-scalable-distributed-deep-rl-with)~.

1. **다중 작업 학습**
IMPALA는 여러 작업을 동시에 학습할 수 있는 능력을 가지고 있습니다. ~[이는 다양한 환경에서의 일반화 성능을 향상시키고, 한 작업에서 얻은 지식을 다른 작업에 적용할 수 있게 합니다3](https://schneppat.com/impala.html)~.

1. **성능 및 데이터 효율성**
IMPALA는 이전의 에이전트들보다 적은 데이터로 더 나은 성능을 달성할 수 있습니다. ~[이는 특히 다중 작업 학습에서 긍정적인 전이를 보여줍니다](https://arxiv.org/abs/1802.01561)[2](https://paperswithcode.com/paper/impala-scalable-distributed-deep-rl-with)~.
이러한 구조적 특징들 덕분에 IMPALA는 대규모 분산 강화 학습 환경에서 매우 효과적으로 작동할 수 있습니다. 더 자세한 내용은 [공식 논문을 참고하세요](https://arxiv.org/abs/1802.01561)
