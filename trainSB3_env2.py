from stable_baselines3 import PPO, A2C, DQN
import os
from monopoly.envs.monopoly_env import MonopolyEnv
import time
import argparse


def run_training(args):
    env = MonopolyEnv(args.num_states, args.dice_size, 
                    args.num_agents, args.max_turns, 
                    args.file_csv)
    env.reset()

    model = PPO(args.model, env, 
                verbose=0, tensorboard_log=logdir)

    TIMESTEPS = args.timesteps
    iters = args.iterations

    position_list = [0, 0]
    file_path = "ownership_data.txt"
    with open(file_path, 'w') as file:
        file.write("")
    
    iteration = 0
    while iteration < iters:
        iteration += 1
        # We can either put env.reset here not; produced same results. But on safer side, we should put
        env.reset()
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
        print

        # This is for debug, as I turned the verbose=0 because it was too verbose. This is better.
        # Potential can do TQDM
        print(f"env.episode_length {env.episode_length}")
        if env.episode_length < 50:
            random_action = env.action_space.sample()

            print(f"Episode")
            print(f"Current_player: {env.current_player.num}")
            print(f"Position before roll: {env.current_player.pos}")
            print("action", random_action)

            obs, reward, done, trunc, info = env.step(random_action)
            print(f"Roll: {env.roll_val}")
            print(f"Position after roll: {env.current_pos}")
            # if reward != 0.:
            print('state', [format(num, '.2f') for num in obs])
            print('reward', reward)
            print(info)
            owner = []
            worths = []
            for city in env.board:
                if city.owner != None:
                    owner_num = city.owner.num
                else:
                    owner_num = 0
                owner.append([city.name, owner_num])
            for player in env.players:
                worths.append([player.num, player.money])
            print("OWNER:",owner)
            print(worths)
            owner_tuple = [(location, value) for location, value in owner]
            position_list[env.current_player.num-1] = env.current_pos
            with open(file_path, 'a') as file:
                file.write(str(owner_tuple)+"\n")
                file.write(str(env.current_player.num)+"\n")
                file.write(str(env.roll_val)+"\n")
                file.write(str(env.actions[random_action])+"\n")
                file.write(str(position_list)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs different kinds of pipelines for MAPF using DT and LLM.')
    
    parser.add_argument('--num_states', default=12, help="Number of squares")
    parser.add_argument('--dice_size', default=6, help="Number of faces on the dices")
    parser.add_argument('--num_agents', default=2, help="Number of total agents")
    parser.add_argument('--max_turns', default=100, help="total number of turns allowed")
    parser.add_argument('--file_csv', default="./assets/city.csv", help="path to City.csv")
    parser.add_argument('--model', default="MlpPolicy", help="model MLP")
    
    parser.add_argument('--timesteps', default=1000, help="timesteps")
    parser.add_argument('--iterations', default=10, help="Iterations")
    parser.add_argument('--num_episodes', default=10, help="numbers episodes we want")
    
    parser.add_argument('--exp_name', default='',
                            help="Experiment name for running multiple runs with same settings and models")

    args = parser.parse_args()

    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    run_training(args)