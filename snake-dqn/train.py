from snake_game_ai import SnakeGame
from agent import Agent

def train():
    record = 0
    agent = Agent()
    game = SnakeGame()
    
    while True:
        state_old = game.get_state()
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        # Train on the single step immediately
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Save to memory buffer
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game over! Train on a batch of past memories
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save() 

            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')
            


if __name__ == '__main__':
    train()