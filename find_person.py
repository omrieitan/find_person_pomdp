import numpy as np
import matplotlib.pyplot as plt
from random import randint


def gauss_2d(mu, sigma, size):
    """
    helper function for generating 2D gaussian kernel
    """
    s = int(size / 2)
    w = np.arange(-s, s + 1)
    x, y = np.meshgrid(w, w)
    d = x * x + y * y
    K = np.exp(-((d - mu) ** 2) / (2.0 * (sigma ** 2)))
    return K / np.sum(K)


def uniform_2d(size, fill_size=None):
    """
    helper function for generating 2D uniform kernel
    """
    if fill_size is None:
        prob = 1 / size ** 2
        return np.full((size, size), prob)
    else:
        prob = 1 / fill_size ** 2
        K = np.full((fill_size, fill_size), prob)
        pad_size = int((size - fill_size) / 2)
        return np.pad(K, [(pad_size,), (pad_size,)], mode='constant')


class find_person_pomdp():
    """
    This class is an implementation for a 'black box'
    for large POMDPs solvers, as described on the paper:
    'Monte-Carlo Planning in Large POMDPs' - by David Silver and Joel Veness
    reference: https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf

    This 'black box' simulate an agent(robot) in a grid world where people are located,
    and his main goal is to guess who is the one person he's looking for.

    Actions:
        the actions space is given by the following functions -
        1. ask() - if the agent is close enough to any of the people(within his kernel),
            he would ask that person whether he is the right person and get an answer that is
            (90% * proximity factor) accurate.
        2. guess(person_index) - at any time, the robot may guess if a person is the one he's looking for.
            the answer is 100% accurate - if guessed correctly, the simulation is over.
            otherwise, the robot knows for sure that person isn't the right one.
        3. move(x,y) - the robot may choose to move in order to get closer to people and get more
            accurate observations over their locations.
        (!) in order to make the actions space finite, the biggest step a robot can take is defined
            by the initial max x and y distances over all people in the grid world.

    Observations:
        the observations of the robot, given by his sensors(LiDAR, Ultrasonic, etc.), are defined by -
        1. self.estimated_locations - an array of estimated location, of every person. sampled from each
            kernel in self.kernels.
            (a) self.locations - an array of the real location, of every person.
            (b) self.kernels - an array of every person's kernel. this kernel represent
                the finite area where the person might be. every entry holds the probability
                it's the person's accurate location. the kernel's entries sum up to 1.
                the center  of the kernel is the real location.
        2. self.estimated_answer - the estimated index of the correct person.
            sampled from self.person_match.
            (a) self.person_match - an array of probabilities for each person to be
                the correct answer. all entries sum up to 1.

    State:
        the real state of the robot at any given time is defined by -
        1. self.answer - the index of the right answer.
        2. self.locations - an array of the real location, of every person.

    (!) in order to make the state and observations spaces finite, we restricted the
        maximum distance the robot can be from the furthest person to be (2 * self.MAX_DIST).
        this is done using the self.valid_move(dx, dy) method.
        trying to make an invalid move would result in -self.MAX_REWARD reward, in order to
        discourage such action.

    Reward:
        defined by self.reward
        move(x,y) actions would result in relatively small negative reward
        the ask() action would result in medium-big negative reward
        guess(person_index) would result in a very big positive reward if correct,
        and equally big negative reward otherwise.
        (!) this reward system is meant to encourage caution behavior - the robot
            should choose to get very close to a person before asking, and he should
            guess only when he's certain of the answer.

    Restrictions:
        1. must define at least 2 people.
        2. only odd sized kernels are supported.
        3. no kernels overlaps are supported.
    """

    # the reward for guessing correctly
    MAX_REWARD = 1000

    def __init__(self, init_locations, kernels_size, kernels_type='gauss', ans=0, init_reward=0.0, person_match=None):
        # check inputs are valid
        if kernels_type != 'uniform' and kernels_type != 'gauss':
            raise ValueError("kernel type must be 'uniform' or 'gauss'.")
        if kernels_size % 2 == 0:
            raise ValueError("kernel size must be an odd number.")
        self.check_loc_overlaps(init_locations, kernels_size)

        self.found_person = 0
        self.time = 0

        self.reward = init_reward

        self.NUM_OF_PEOPLE = len(init_locations)
        # if not given, person_match would be uniform distribution.
        if person_match is not None and len(person_match) == self.NUM_OF_PEOPLE:
            self.person_match = person_match
        else:
            self.person_match = np.full(self.NUM_OF_PEOPLE, 1.0 / self.NUM_OF_PEOPLE)

        # kernel centers
        self.locations = init_locations
        # the distance to the furthest person times 2
        self.MAX_DIST = self.get_max_distance()

        # kernels that define the people's location distribution
        if kernels_type == 'uniform':
            self.kernels = [uniform_2d(kernels_size) for _ in range(self.NUM_OF_PEOPLE)]
        else:
            self.kernels = [gauss_2d(0, kernels_size, kernels_size) for _ in range(self.NUM_OF_PEOPLE)]
        self.kernels_type = kernels_type

        # if not defined or invalid, the answer would be randomly selected
        self.answer = ans - 1
        if self.answer < 0 or self.answer >= self.NUM_OF_PEOPLE:
            self.answer = randint(0, self.NUM_OF_PEOPLE - 1)

        self.KERNEL_SIZE = int(kernels_size / 2)

        # generate actions array of all possible actions, according to the given input
        self.actions = []
        # actions strings for debugging
        self.actions_str = []
        self.generate_actions()

        self.update_loc(0, 0)
        self.estimated_locations = np.zeros(self.locations.shape, dtype=int)
        self.sample_locations()

        # estimate who is the right person
        self.estimated_answer = np.random.choice(np.arange(self.NUM_OF_PEOPLE), 1, p=self.person_match)

        self.observations = [self.estimated_answer, self.estimated_locations]
        self.state = [self.answer, self.locations]

    @staticmethod
    def check_loc_overlaps(init_locations, kernels_size):
        """
        A helper function for __init__ to check for kernels overlaps.
        """
        for i in range(len(init_locations)):
            for j in range(i + 1, len(init_locations)):
                if abs(init_locations[i][0] - init_locations[j][0]) < kernels_size and abs(
                        init_locations[i][1] - init_locations[j][1]) < kernels_size:
                    raise ValueError("kernel-{0} and kernel-{1} are overlapping.".format(i, j))

    def sample_locations(self):
        """
        A helper function, used to update the estimated location of each person
        by sampling it's location from the person's kernel.
        """
        indices = np.arange(self.kernels[0].size)
        for i, kernel in enumerate(self.kernels):
            idx_1d = np.random.choice(indices, 1, p=np.ravel(kernel))
            y = int(idx_1d / kernel.shape[0]) + self.locations[i][0] - self.KERNEL_SIZE
            x = idx_1d % kernel.shape[1] + self.locations[i][1] - self.KERNEL_SIZE
            self.estimated_locations[i] = [y, x]

    def generate_actions(self):
        """
        generate self.actions array of all possible actions, according to the given input.
        also generate corresponding self.actions_str for debugging.
        """
        max_x = 0
        max_y = 0
        for p in self.locations:
            curr_x = p[0]
            curr_y = p[1]
            if curr_x > max_x:
                max_x = curr_x
            if curr_y > max_y:
                max_y = curr_y

        self.actions.append((self.ask, []))
        self.actions_str.append("ask()")
        for i in range(1, self.NUM_OF_PEOPLE + 1):
            self.actions.append((self.guess, [i]))
            self.actions_str.append("guess({})".format(i))
        i = self.NUM_OF_PEOPLE + 1
        for x in range(-max_x, max_x + 1):
            for y in range(-max_y, max_y + 1):
                self.actions.append((self.move, [x, y]))
                self.actions_str.append("move({0},{1})".format(x, y))
                i += 1

    def print_actions(self, index=None):
        """
        if index is specified -
            print the string of the corresponding action.
        else -
            print all possible actions, with their corresponding index.
        format: <index> - <action string>
        """
        if index is not None:
            print("{0} - {1}".format(index, self.actions_str[index]))
        else:
            for i, action in enumerate(self.actions_str):
                print("{0} - {1}".format(i, action))

    def get_max_distance(self):
        """
        :return: the distance to the furthest person times 2.
        """
        dist = 0
        for p in self.locations:
            curr = (abs(p[0]) + abs(p[1])) * 2
            if curr > dist:
                dist = curr
        return dist

    def valid_move(self, dx, dy):
        """
        A helper for move(dx,dy).
        to make sure the robot can get further than  2 * self.MAX_DIST
        from the furthest person, in order to make the state and observations
        spaces finite.
        (!) trying to make an invalid move would result in -self.MAX_REWARD reward,
            in order to discourage such action.
        """
        for i, p in enumerate(self.locations):
            if abs(p[0] - dx) + abs(p[1] - dy) > 2 * self.MAX_DIST:
                print("Invalid move, too far from person {}!".format(i))
                self.reward -= self.MAX_REWARD
                return False
        return True

    def move(self, dx, dy):
        """
        The move(x,y) action
        :param dx:(int) how much to move in the x direction
        :param dy:(int) how much to move in the y direction

        observation:
            change the robot's observation according to how close it gets
            to people - the closer, the more accurate sensors reading are
        reward:
            add a small negative reward, while taking into consideration proximity bonus.
            proximity bonus is given for moving in the direction of a possible
            correct answer. (still total reward remains negative)
        state:
            self.locations is changed according to the dx,dy movement.
        """
        self.time += 1
        if not self.valid_move(dx, dy):
            print("try again, with different inputs.")
            return

        curr_reward = -(np.abs(dx) + np.abs(dy))
        curr_reward += self.proximity_bonus(dx, dy)
        self.reward += curr_reward

        self.update_loc(dx, dy)
        self.sample_locations()

    def update_loc(self, dx, dy):
        """
        A helper function for move(x,y).
        :param dx:(int) how much to move in the x direction
        :param dy:(int) how much to move in the y direction

        update:
            1. self.locations - kernels centers
            2. self.kernels - the distribution of people's locations.
                the closer we get, the more clear a person's location gets.
        """
        for i, p in enumerate(self.locations):
            p[0] -= dx
            p[1] -= dy
            k_dist = self.KERNEL_SIZE * 2 + 1
            if self.kernels_type == 'uniform':
                size = int(k_dist * (abs(p[0]) + abs(p[1])) / self.MAX_DIST)
                if size % 2 == 0:
                    size += 1  # can't allow even sized kernels
                if size < 1:
                    size = 1
                elif size > k_dist:
                    size = k_dist
                self.kernels[i] = uniform_2d(k_dist, fill_size=size)
            else:
                sigma = ((abs(p[0]) + abs(p[1])) / self.MAX_DIST * k_dist) + 0.1
                self.kernels[i] = gauss_2d(0, sigma, k_dist)

    def proximity_bonus(self, x, y):
        """
        An helper function for move(x,y).
        :param x:(int) how much to move in the x direction
        :param y:(int) how much to move in the y direction
        :return: a small positive reward for getting closer to possibly correct answer/s.
        """
        delta = [np.abs(px) - np.abs(px - x) + np.abs(py) - np.abs(py - y) for (px, py) in self.locations]
        return np.sum(delta * self.person_match)

    def ask(self):
        """
        The ask() action.
        check if one of the people is close enough to ask if he is the right person,
        and update self.person_match according to the given answer,
        while taking into consideration it's 90% * (the kernel's center probability) accurate.

        observation:
            self.estimated_answer is sampled from self.person_match after this action.
        reward:
            adds negative reward equal to -(MAX_REWARD / 10).
        state:
            is not affected by this action.
        """
        self.time += 1
        self.reward -= self.MAX_REWARD / 10
        idx = self.get_close_person()

        if idx > -1:
            chance = self.kernels[idx][self.KERNEL_SIZE, self.KERNEL_SIZE]
            if idx == self.answer:  # the right person would say it's him with 90% chance
                self.handle_ask(idx, [chance * 0.9, 1 - chance * 0.9])
            else:  # the wrong person would say it's him with 10% chance
                self.handle_ask(idx, [chance * 0.1, 1 - chance * 0.1])
        # else nothing would happen except negative reward

        # estimate who is the right person
        self.estimated_answer = np.random.choice(np.arange(self.NUM_OF_PEOPLE), 1, p=self.person_match)

    def get_close_person(self):
        """
        An helper for ask().
        :return: the index of the person the robot's is close to (within his kernel)
            if doesn't exist, return -1.
        """
        for i, [x, y] in enumerate(self.locations):
            if abs(x) <= self.KERNEL_SIZE and abs(y) <= self.KERNEL_SIZE:
                return i
        return -1

    def handle_ask(self, idx, prob):
        """
        An helper for ask().
        :param idx:(int) a person's index.
        :param prob:(float) the probability the robot would get an accurate response.

        update self.person_match according to the given response.
        """
        if np.random.choice([1, 0], 1, p=prob):  # yes/no, 90% chance to understand answer
            self.person_match[idx] += (1 - self.person_match[idx]) * 0.9
            self.person_match[:idx] *= 0.1
            self.person_match[idx + 1:] *= 0.1
        else:
            rest = np.sum(self.person_match[:idx]) + np.sum(self.person_match[idx + 1:])
            for i in range(self.NUM_OF_PEOPLE):
                if i != idx:
                    self.person_match[i] += self.person_match[i] * self.person_match[idx] * 0.9 / rest
            self.person_match[idx] *= 0.1

    def guess(self, idx):
        """
        The guess(person_index) action
        :param idx:(int) a person's index.

        if guessed correctly, self.person_match[idx] = 1 and all other would be 0.
        otherwise, self.person_match[idx] = 0, and it's previous prob would be divided
        between the other possible answers(people).

        observation:
            self.estimated_answer is sampled from self.person_match after this action.
        reward:
            if guessed correctly -
                add MAX_REWARD
            else -
                add -MAX_REWARD
        state:
            is not affected by this action.
        """
        self.time += 1
        if self.answer == idx - 1 and (not self.found_person):
            self.reward += self.MAX_REWARD
            self.person_match[idx - 1] = 1
            self.person_match[:idx - 1] = 0
            self.person_match[idx:] = 0
            self.found_person = 1  # task complete, stop taking actions!
        else:
            self.reward -= self.MAX_REWARD
            rest = np.sum(self.person_match[:idx - 1]) + np.sum(self.person_match[idx:])
            for i in range(self.NUM_OF_PEOPLE):
                if i != idx - 1:
                    self.person_match[i] += self.person_match[i] * self.person_match[idx - 1] / rest
            self.person_match[idx - 1] = 0

        # estimate who is the right person
        self.estimated_answer = np.random.choice(np.arange(self.NUM_OF_PEOPLE), 1, p=self.person_match)

    def visualize(self):
        """
        A tool for debugging.
        this function is used to visualize observations and current state of the robot.
        """
        fig = plt.figure()
        plot_size = 1
        while plot_size ** 2 < self.NUM_OF_PEOPLE + 1:
            plot_size += 1
        for i in range(1, self.NUM_OF_PEOPLE + 1):
            fig.add_subplot(plot_size, plot_size, i)
            plt.imshow(self.kernels[i - 1], cmap='hot')
            plt.title(
                "person {0}\nreal location: ({1}, {2})\nestimated location: ({3}, {4})".format(i,
                                                                                               self.locations[i - 1][0],
                                                                                               self.locations[i - 1][1],
                                                                                               self.estimated_locations[
                                                                                                   i - 1, 0],
                                                                                               self.estimated_locations[
                                                                                                   i - 1, 1]))
        fig.add_subplot(plot_size, plot_size, self.NUM_OF_PEOPLE + 1)
        plt.pie(self.person_match, labels=['person {}'.format(i + 1) for i in range(self.NUM_OF_PEOPLE)],
                autopct='%1.1f%%')
        plt.title('score: {0:.3f}\ntime: {1}\nestimated answer: person {2}'.format(self.reward, self.time,
                                                                                   self.estimated_answer[0] + 1))
        plt.show()
