"""Train a PPO agent."""

import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import rich_click as click
import torch
from rich.console import Console
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ...gym.game import KnightGame, KnightObsWrapper
from ...gym.game_example import get_env
from .agent import SAVE_OPTIMISER, SAVE_UPDATES, SAVE_WEIGHTS, TF_BOARD, TORCH_PARAMS, ActorCriticNetwork, obs_to_torch

EXP_NAME = "train-ppo"
ENV_NAME = "simple"
SEED = 42
TRACK = False
WANDB_PROJECT_NAME = "knight-five"
WANDB_ENTITY = None
RESUME_TRAIN = False
RUN_DIR = None
WEIGHTS_FREQ = 1
TOTAL_TIMESTEPS = 500_000
NUM_ENVS = 4
NUM_STEPS = 30
LEARNING_RATE = 2.5e-4
ANNEAL_LR = True
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
NORM_ADV = True
CLIP_COEF = 0.2
ENT_COEF = 0.01
CLIP_VLOSS = True
VF_COEF = 0.5
TARGET_KL = None


def make_env(env_name: str, seed: int, idx: int, run_name: str):
    """Function factory to create a new env."""
    base_env = get_env(env_name=env_name)

    def create_env():
        env = KnightObsWrapper(KnightGame(board=base_env.board, goal=base_env.goal, start=base_env.start))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return create_env


@click.command(context_settings={"show_default": True})
@click.option("-n", "--exp-name", type=str, default=EXP_NAME, help="name of this experiment")
@click.option("-e", "--env-name", type=str, default=ENV_NAME, help="environment to use")
@click.option("-s", "--seed", type=int, default=SEED, help="seed of the experiment")
@click.option(
    "-t",
    "--track",
    is_flag=True,
    default=TRACK,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
@click.option("--wandb-project-name", type=str, default=WANDB_PROJECT_NAME, help="wandb's project name")
@click.option("--wandb-entity", type=str, default=WANDB_ENTITY, help="entity (team) of wandb's project")
@click.option(
    "--resume-train",
    is_flag=True,
    default=RESUME_TRAIN,
    help="toggle to resume training from a previously saved model and optimiser state",
)
@click.option("--run-dir", type=str, default=RUN_DIR, help="directory from where to resume the training")
@click.option(
    "--weights-freq", type=int, default=WEIGHTS_FREQ, help="weight saving frequency, as every `n` optimisation steps"
)
@click.option("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, help="total timesteps of the experiments")
@click.option("--num-envs", type=int, default=NUM_ENVS, help="number of parallel game environments")
@click.option(
    "--num-steps", type=int, default=NUM_STEPS, help="number of steps to run in each environment per policy rollout"
)
@click.option("--learning-rate", type=float, default=LEARNING_RATE, help="learning rate of the optimizer")
@click.option(
    "--anneal-lr",
    is_flag=True,
    default=ANNEAL_LR,
    show_default=True,
    help="toggle learning rate annealing for policy and value networks",
)
@click.option("--max-grad-norm", type=float, default=MAX_GRAD_NORM, help="maximum norm for the gradient clipping")
@click.option("--update-epochs", type=int, default=UPDATE_EPOCHS, help="number of epochs to update the policy")
@click.option("--num-minibatches", type=int, default=NUM_MINIBATCHES, help="number of mini-batches")
@click.option("--gamma", type=float, default=GAMMA, help="discount factor gamma")
@click.option("--gae-lambda", type=float, default=GAE_LAMBDA, help="lambda for the general advantage estimation")
@click.option(
    "--norm-adv",
    is_flag=True,
    default=NORM_ADV,
    show_default=True,
    help="toggle advantages normalization",
)
@click.option("--clip-coef", type=float, default=CLIP_COEF, help="surrogate clipping coefficient")
@click.option("--ent-coef", type=float, default=ENT_COEF, help="coefficient of the entropy")
@click.option(
    "--clip-vloss",
    is_flag=True,
    default=True,
    show_default=True,
    help="toggle whether or not to use a clipped loss for the value function, as per the paper.",
)
@click.option("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
@click.option("--target-kl", type=float, default=None, help="target KL divergence threshold.")
@click.pass_context
def train(
    ctx: click.Context,
    exp_name: str = EXP_NAME,
    env_name: str = ENV_NAME,
    seed: int = SEED,
    track: bool = TRACK,
    wandb_project_name: str | None = WANDB_PROJECT_NAME,
    wandb_entity: str | None = WANDB_ENTITY,
    resume_train: bool = RESUME_TRAIN,
    run_dir: str | None = RUN_DIR,
    weights_freq: int = WEIGHTS_FREQ,
    total_timesteps: int = TOTAL_TIMESTEPS,
    num_envs: int = NUM_ENVS,
    num_steps: int = NUM_STEPS,
    learning_rate: float = LEARNING_RATE,
    anneal_lr: bool = ANNEAL_LR,
    max_grad_norm: float = MAX_GRAD_NORM,
    update_epochs: int = UPDATE_EPOCHS,
    num_minibatches: int = NUM_MINIBATCHES,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
    norm_adv: bool = NORM_ADV,
    clip_coef: float = CLIP_COEF,
    ent_coef: float = ENT_COEF,
    clip_vloss: bool = CLIP_VLOSS,
    vf_coef: float = VF_COEF,
    target_kl: float | None = TARGET_KL,
) -> None:
    """Train a PPO agent."""
    all_args = {k: v for k, v in locals().items() if k != "ctx"}
    run_name = f"{env_name}__{exp_name}__{seed}__{int(time.time())}"

    if track:
        import wandb

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            config=all_args,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    console = Console()
    device = "cpu"

    # Monitoring
    if run_dir is not None:
        run_dir = Path(run_dir).resolve(strict=True)
    else:
        cwd = ctx.obj["cwd"]
        run_dir: Path = cwd / "runs" / run_name
        console.print(f"📁 create saving dir for the run at {run_dir}", style="gold3")
        run_dir.mkdir(parents=True)
    weights_dir = run_dir / TORCH_PARAMS
    weights_dir.mkdir(exist_ok=True)
    tf_dir = run_dir / TF_BOARD
    tf_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(tf_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in all_args.items()])),
    )

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Env setup
    envs = gym.wrappers.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv(
            [make_env(env_name=env_name, seed=seed, idx=i, run_name=run_name) for i in range(num_envs)]
        )
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # custom method to get shape
    obs_shapes = envs.envs[0].get_obs_shape()
    action_shape = envs.single_action_space.n

    agent = ActorCriticNetwork(conv_dim=obs_shapes["conv"], fc_dim=obs_shapes["fc"], action_dim=action_shape).to(device)

    # Setup optimisation
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    num_updates = total_timesteps // batch_size
    weight_update_idx = 1

    # Resume training if needed
    if resume_train:
        if weights_dir is None:
            raise ValueError("you cannot resume training without passing a weight directory")
        last_dir = max(weights_dir.iterdir(), key=lambda s: int(s.name.split("-")[1]))
        agent.load_state_dict(torch.load(last_dir / SAVE_WEIGHTS))
        optimizer.load_state_dict(torch.load(last_dir / SAVE_OPTIMISER))
        learning_rate = optimizer.param_groups[0]["lr"]
        console.print(f"⚠️ training is resumed, learning is set to previous learning rate {learning_rate}")
        weight_update_idx += int(last_dir.name.split("-")[1]) + 1

    # ALGO Logic: Storage setup
    storage_size = (num_steps, num_envs)
    obs_conv = torch.zeros((*storage_size, *obs_shapes["conv"])).to(device)
    obs_fc = torch.zeros((*storage_size, *obs_shapes["fc"])).to(device)
    action_mask = torch.zeros((*storage_size, *obs_shapes["action_mask"]), dtype=torch.bool).to(device)
    actions = torch.zeros((*storage_size, *envs.single_action_space.shape)).to(device)
    logprobs = torch.zeros(storage_size).to(device)
    rewards = torch.zeros(storage_size).to(device)
    dones = torch.zeros(storage_size).to(device)
    values = torch.zeros(storage_size).to(device)

    # Start the training loop
    global_step = 0
    start_time = time.time()
    next_obs, next_info = envs.reset()
    next_obs = obs_to_torch(next_obs, device=device)
    next_done = torch.zeros(num_envs).to(device)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect steps from the vectorized environment in the storage tensors
        # env are not automatically reset at the end of the collection loop,
        # allowing to continue episode which are not done yet
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs_conv[step] = next_obs["conv"]
            obs_fc[step] = next_obs["fc"]
            action_mask[step] = next_obs["action_mask"]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():  # inference, no need for grad
                action, logprob, _, value = agent.get_action_entropy_value(
                    conv_obs=next_obs["conv"],
                    fc_obs=next_obs["fc"],
                    action_mask=next_obs["action_mask"],
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = obs_to_torch(next_obs, device=device)
            next_done = torch.tensor(done, dtype=torch.int).to(device)

            if "episode" in info:
                item = info["episode"]
                mask = info["_episode"]
                episodic_returns = item["r"][mask]
                episodic_length = item["l"][mask]
                episodic_minutes = (d["minutes"] for d in info["final_info"][mask])
                for r, l, m in zip(episodic_returns, episodic_length, episodic_minutes):
                    style = "dark_red" if r < 0 else "dark_green"
                    console.print(f"global_step={global_step}, episodic_return={r}", style=style, highlight=False)
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)
                    writer.add_scalar("charts/episodic_minutes", m, global_step)
                    # break

        # steps collection is over. might the case we still did not reach the end of episode
        # this is why storage keeps track of steps for each env and did not mix them up
        # bootstrap value if not done
        with torch.no_grad():  # inference, no need for grad
            # next_obs is the last obs from the collection loop
            next_value = agent.get_value(conv_obs=next_obs["conv"], fc_obs=next_obs["fc"]).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch -> don't care from which environment the transition is coming from
        b_obs_conv = obs_conv.reshape((-1, *obs_shapes["conv"]))
        b_obs_fc = obs_fc.reshape((-1, *obs_shapes["fc"]))
        b_action_mask = action_mask.reshape((-1, *obs_shapes["action_mask"]))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_entropy_value(
                    conv_obs=b_obs_conv[mb_inds],
                    fc_obs=b_obs_fc[mb_inds],
                    action_mask=b_action_mask[mb_inds],
                    action=b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # debug parameters to check if policy is changing
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        # save network and optimiser state for later reuse
        if weight_update_idx % weights_freq == 0:
            save_dir = weights_dir / SAVE_UPDATES.format(weight_update_idx=weight_update_idx)
            save_dir.mkdir()
            torch.save(agent.state_dict(), save_dir / SAVE_WEIGHTS)
            torch.save(optimizer.state_dict(), save_dir / SAVE_OPTIMISER)
        weight_update_idx += 1

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        console.print("SPS:", int(global_step / (time.time() - start_time)), style="purple")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
