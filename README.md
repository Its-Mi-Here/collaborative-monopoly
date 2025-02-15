# Monopoly++: An Environment for Collaborative Monopoly

A small Gym-compatible Monopoly environment for experimentation with collaborative Multi-agent Reinforcement learning (MARL). Our modified version of Monopoly allows 2 players to collaborate through different features like rent-waivers and loaning. This environment can be used to observe the collaborative behaviors of RL agents & design reward specifications for strategies which optimize team goals.

This Repository is the implementation of this academic [project](https://drive.google.com/file/d/1K7ZJ57sMuA3jk0KWG4UC_EZ6ENHZreM0/view?usp=sharing)

## Abstract
Despite the vast research in the field of fully collaborative and fully adversarial Multi-agent Reinforcement Learning (MARL), there’s a relative scarcity in mixed environments. The exploration of collaborative decision-making in a competitive environment has been observed, however, it has been implemented such that team members can coordinate their strategies before the beginning of the game. To bring novelty to this, we introduce in-game collaboration in a simpler Monopoly environment - Monopoly++. This game provides an adversarial environment with a discrete action space that also mimics real-world interactions. By delving into the complex interactions between two players working collaboratively, the project studies the nuanced tactics that contribute to success in a Monopoly game and contribute to a broader understanding of collaborative strategies in mixed environments. We also release an open-source custom Monopoly++ environment facilitating future research into these environments.

## Demo
The below video is a demonstration of our environment. It also shows how the trained RL agents are collaborating and making one of the agents win the game(own all the tiles) when playing against static agents.

<!-- https://github.com/Its-Mi-Here/collaborative-monopoly/blob/main/assets/Monopoly3.mp4
 -->
 https://github.com/user-attachments/assets/880ebebe-2aee-42e3-8f8c-746aef43fe9b
