# dual-blind
This is the open source code of paper "CMA: Collaborative Microservice Autoscaling Based on
Reinforcement Learning in Edge-Cloud Environment"

Directory Structure:
"algos" and "onpolicy" folders store algorithm-related files.
runner is the entry point for functions. Within it, GMPERunner().train() is used to train the algorithm, and GMPERunner().eval_cma() is used to evaluate the algorithm.
Potential Issues:
1. Incorrect directory structure.
2. Issues caused by path changes.