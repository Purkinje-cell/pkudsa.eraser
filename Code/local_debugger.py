from interaction import *
import my_robot_2 as player1
import my_robot as player2

print("Initializing game...")
game1 = Game_play(player2, player1, seed=1)
game2 = Game_play(player1, player2, seed=1)

print("Game running...")
game1.start_game()
game2.start_game()

print("Saving replay...")
game1.save_log('replay1_1.json')
game2.save_log('replay2_1.json')
print("Done")
