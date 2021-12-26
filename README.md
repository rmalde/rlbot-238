# Using Curriculum Training and Negative Rewards to Train a Rocket League Kickoff Bot
CS 238 Final Project

By: Ronak Malde and Arjun Karanam

Video: https://youtu.be/vx8clBZe5Yc 

Link to Paper: https://github.com/rmalde/rlbot-238/blob/main/Final%20Paper.pdf

Abstract:

This paper seeks to use Reinforcement Learning in order to play a game known as Rocket League, where players play soccer in a virtual environment using cars. Rocket League's large action spaces, state spaces, and delayed reward (scoring a goal) make it a challenging yet fruitful endeavor. By using the Rocket League Gym environment (similar to OpenAI's Gym Environment), we train a bot to reliably do in the kickoff phase of the game (the beginning phase where the player is randomly placed on the field and has the ability to make a goal). In addition to current approaches in Rocket League literature, we tried Negative Reward functions as well as curriculum learning. While curriculum learning failed to show improved results, Negative Reward Functions allowed us to reach kickoff metrics that are comparable to a human player and current leading hard-coded bots. In the future, we would like to take other phases of the game (such as defending a goal, attacking a goal, etc.) and attempt RL methods with Negative rewards to one day train a bot to play the game in its entirety. 
