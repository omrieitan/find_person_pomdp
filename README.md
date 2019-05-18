# Find Person - a black box for large POMDPs
This class is an implementation for a 'black box'
  for large POMDPs solvers, as described on the paper:
  ['Monte-Carlo Planning in Large POMDPs' - by David Silver and Joel Veness](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)

  This 'black box' simulate an agent(robot) in a grid world where people are located,
  and his main goal is to guess who is the one person he's looking for.

  * require _python 3.5_ or higher to run.

## Description
### Actions:
  the actions space is given by the following functions -
  1. _ask()_ - if the agent is close enough to any of the people(within his kernel),
      he would ask that person whether he is the right person and get an answer that is
      (90% * proximity factor) accurate.
  2. _guess(person_index)_ - at any time, the robot may guess if a person is the one he's looking for.
      the answer is 100% accurate - if guessed correctly, the simulation is over.
      otherwise, the robot knows for sure that person isn't the right one.
  3. _move(x,y)_ - the robot may choose to move in order to get closer to people and get more
      accurate observations over their locations.
  * in order to make the actions space finite, the biggest step a robot can take is defined
      by the initial max x and y distances over all people in the grid world.

### Observations:
the observations of the robot, given by his sensors(LiDAR, Ultrasonic, etc.), are defined by -
1. _self.estimated_locations_ - an array of estimated location, of every person. sampled from each
    kernel in self.kernels.
    * _self.locations_ - an array of the real location, of every person.
    * _self.kernels_ - an array of every person's kernel. this kernel represent
        the finite area where the person might be. every entry holds the probability
        it's the person's accurate location. the kernel's entries sum up to 1.
        the center  of the kernel is the real location.
2. _self.estimated_answer_ - the estimated index of the correct person.
    sampled from _self.person_match_.
    * _self.person_match_ - an array of probabilities for each person to be
        the correct answer. all entries sum up to 1.

### State:
the real state of the robot at any given time is defined by -
1. _self.answer_- the index of the right answer.
2. _self.locations_ - an array of the real location, of every person.



### Reward:
defined by _self.reward_
_move(x,y)_ actions would result in relatively small negative reward
the _ask()_ action would result in medium-big negative reward
_guess(person_index)_ would result in a very big positive reward if correct,
and equally big negative reward otherwise.
* this reward system is meant to encourage caution behavior - the robot
    should choose to get very close to a person before asking, and he should
    guess only when he's certain of the answer.

## Additional notes
* in order to make the state and observations spaces finite, we restricted the
  maximum distance the robot can be from the furthest person to be _(2 * self.MAX_DIST)_.
  this is done using the _self.valid_move(dx, dy)_ method.
  trying to make an invalid move would result in _-self.MAX_REWARD_ reward, in order to discourage such action.
* further documentation and explanations can be found in the code.
* a human interface for the __find_person_pomdp__ class is provided in the main.py file.
  it's purpose is to help you debug and understand the functionality
  of the black box.

## Restrictions:
1. must define at least 2 people.
2. only odd sized kernels are supported.
3. no kernels overlaps are supported.
