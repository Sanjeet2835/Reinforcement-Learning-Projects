import torch
import pygame
from snake_game_ai import SnakeGame
from model import Linear_QNet

def play():
    # 1. Initialize the game environment
    game = SnakeGame()
    
    # 2. Initialize the exact same brain architecture we used for training
    # (7 inputs, 256 hidden nodes, 3 outputs)
    model = Linear_QNet(7, 256, 3) 
    
    # 3. LOAD THE TRAINED BRAIN!
    # We use weights_only=True for security best practices when loading PyTorch models
    model.load_state_dict(torch.load('./model/model.pth', weights_only=True))
    
    # Put the model in evaluation mode (turns off things like dropout/batchnorm if we had them)
    model.eval() 
    
    print("Trained brain loaded! Starting the game...")

    while True:
        # Step A: Get the current state
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float)
        
        # Step B: The Brain predicts the best move (NO RANDOMNESS, NO EPSILON)
        # We use torch.no_grad() because we don't need to calculate gradients for training anymore.
        # This makes inference much faster and saves memory.
        with torch.no_grad():
            prediction = model(state_tensor)
            move_idx = torch.argmax(prediction).item()
        
        # Convert to our [1,0,0] action format
        final_move = [0, 0, 0]
        final_move[move_idx] = 1
        
        # Step C: Perform the move in the game
        reward, done, score = game.play_step(final_move)
        
        # Optional: If you removed self.clock.tick() from your game to train faster, 
        # add a small delay here so you can actually watch it play at normal speed!
        pygame.time.delay(30) 
        
        if done:
            print(f"Game Over! Final Score: {score}")
            game.reset()
            # Pause for a second before the next game starts
            pygame.time.delay(1000) 

if __name__ == '__main__':
    play()