from find_person import find_person_pomdp
import numpy as np

if __name__ == '__main__':
    """
    This is a human interface for the 'find_person_pomdp' class.
    it's purpose is to help you debug and understand the functionality
    of the black box.

    it is recommended to try different initial inputs, and see the result reward
    for different policies.
    """
    init_loc = np.array([[-3, 6], [10, 3], [3, -6]], dtype=int)
    kernels_size = 7
    ans = 2
    action_seq = [103, 102, 106]
    test = find_person_pomdp(init_loc, kernels_size)

    done = 0
    test.visualize()
    while not done:
        curr_input = input("""Choose action:
        1 - move
        2 - ask
        3 - guess
        4 - print all actions
        5 - action by index
        6 - use action sequence
        7 - exit
        """)
        if curr_input == "1":
            x = int(input("input x"))
            y = int(input("input y"))
            test.move(x, y)
        elif curr_input == "2":
            test.ask()
        elif curr_input == "3":
            guess = int(input("what is your guess? <1-{}>".format(test.NUM_OF_PEOPLE)))
            if guess < 1 or guess > test.NUM_OF_PEOPLE:
                print("invalid guess!")
            else:
                test.guess(guess)
                if test.found_person:
                    print("Success, person is found!!!")
                    done = 1
                else:
                    print("wrong guess :(")
        elif curr_input == "4":
            test.print_actions()
        elif curr_input == "5":
            idx = int(input("choose index between 0-{}".format(len(test.actions))))
            if idx < 0 or idx >= len(test.actions):
                print("invalid index!")
            else:
                print("action used: {}".format(test.actions_str[idx]))
                test.actions[idx][0](*test.actions[idx][1])
        elif curr_input == "6":
            for i in action_seq:
                test.actions[i][0](*test.actions[i][1])
        elif curr_input == "7":
            done = 1
        else:
            print("{} is invalid, try again".format(curr_input))
        test.visualize()
