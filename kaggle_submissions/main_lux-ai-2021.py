from stable_baselines3 import A2C  # pip install stable-baselines3
from stable_baselines3 import PPO
from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from agent_policy import AgentPolicy
from luxai2021.env.agent import Agent

if __name__ == "__main__":
    """
    This is a kaggle submission, so we don't use command-line args
    and assume the model is in model.zip in the current folder.
    """
    # Tool to run this against itself locally:
    # "lux-ai-2021 --seed=100 main_lux-ai-2021.py main_lux-ai-2021.py --maxtime 10000"

    # Run a kaggle submission with the specified model
    configs = LuxMatchConfigs_Default

    # Load the saved model
    #model_id = 5403
    #total_steps = int(48e6)

    model = A2C.load("rl_model_1_9000000_steps.zip")
    
    # Create a kaggle-remote opponent agent
    opponent = Agent()

    # Create a RL agent in inference mode
    player = AgentPolicy(mode="inference", model=model)

    # Run the environment
    env = LuxEnvironment(configs, player, opponent)
    env.reset()  # This will automatically run the game since there is
    # no controlling learning agent.
