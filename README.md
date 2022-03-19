# Improving Zero-shot Generalization in Offline Reinforcement Learning using Generalized Similarity Functions
Code for GSF learning in offline Procgen.

**Note**: The repo is under construction right now, some experiments might still be changed/ added.

Since the dataset is very large due to operating on pixel observations, we provide a way to generate it from pre-trained PPO checkpoints instead of hosting 1Tb+ of data.


## Instructions

1. Clone the repo
2. Either train a PPO agent from scratch on 200 levels (see here: [here](https://github.com/bmazoure/ppo_jax)), or download provided PPO checkpoints (same repo link). TLDR, you can run `python train_ppo.py --env_name=bigfish` in the current repo to do so.
3. Run `python evaluate_ppo.py --dataset_dir <output_dir> --shards <num_files_per_env> --timesteps <num_frames_to_save> --obs_type rgb --model_dir=<path_to_PPO_checkpoints_above>`.
This will generate `obs_X.npy, action_X.npy, reward_X.npy, done_X.npy` arrays, where `X` goes from 1 to `n_shards`.
4. You can then work on these NumPy arrays in the classical offline setting, or even online setting (update to GSF shows they work even in the online setting).
5. If you'd like to train a CQL agent on the new dataset, run `python train_gsf_offline.py --cql_reg=1.0`. Don't forget to change the dataset path in the file!

To cite:
```
@article{mazoure2021improving,
  title={Improving Zero-shot Generalization in Offline Reinforcement Learning using Generalized Similarity Functions},
  author={Mazoure, Bogdan and Kostrikov, Ilya and Nachum, Ofir and Tompson, Jonathan},
  journal={arXiv preprint arXiv:2111.14629},
  year={2021}
}
```