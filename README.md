# Model Predictive Minimal Intervention Shared Control (MPMI-SC)

This repository includes the author's implementation of the shared control paradigm presented in "Highly Parallelized Data-driven MPC for Minimal Intervention Shared Control". This work will be presented at RSS 2019 in Freiburg, Germany.  This paradigm ahderes to the following three ideals

1. safety is paramount,
2. the user has no explicit task goal, and
3. the autonomy should exert as little influence as possible.

A detailed description of the shared control algorithm (e.g. modeling technique, the safety computation, etc...) can be found in the corresponding paper.

### Simulated Environments for Evaluation

The shared control algorithm is implemented for two simulated experiental environments: a **balance bot** and a **race car** (see below).

<img src=./images/balance-bot.png alt="Balance Bot" width="400px"/> <img src=./images/race-car.png alt="Race Car" width="400px"/>

In the **balance bot** environment, the system is considered safe so long as the beam does not collide with the ground.  In the **race car** environment the system is considered safe so long as the car remains on the road.

To control the **balance bot**, one can provide desired velocities by pressing left or right on the right joystick on a PS3 remote.  This value can range from [-1 to 1]. See an example of this interface below on the left.

To control the **race car**, one can produce positive acceleration (gas: [0 to 1]) by pressing up on the right joystick and negative acceleration (break: [-1 to 0]) by pressing down on the right joystick.  The operator can also change the heading of the car by pressing left and right on the joystick [-1 to 1].  See an example image of this interface below on the right.

The left and right joysticks switch functionality if the operator is left handed.

<img src=./images/ps3-balance-bot.jpg alt="Balance Bot Controller" width="300px"/> <img src=./images/ps3-race-car.jpg alt="Race Car Controller" width="300px"/>

### Description of packages

 * **gym/** - Our simulated environments based off of OpenAI's gym.
 * **balance_bot_shared_control/** - Our implementation of MPMI-SC in the balance bot environment.  This environment is implemented in PyBullet.
 * **race_car_shared_control/** - Our implementation of MPMI-SC in the race car environment.  This environment is implemented in PyBox2D.


### Initialize codebase

To run this code on your own computer, copy the **mpmi_sc** directory to a location of your choosing, then...
1. Open terminal and build ROS workspace
```Shell
  cd mpmi_sc/
  catkin_make
  ```

### Run system
1. **Connect PS3 Controller**: To connect your PS3 joystick to your computer, open a terminal and run
```Shell
  sudo sixad -s
  ```
2. **Start chosen simulated environment**:
 * Balance Bot
 ```Shell
  roslaunch balance_bot_shared_control system.launch
  ```
 * Race Car
 ```Shell
  roslaunch race_car_shared_control system.launch
  ```

* These commands allow you to run the system under **user only control** and **shared control**.  This choice (and other parameters) can be set in the **config.yaml** file in each package (i.e. src/balance_bot_shared_control and src/race_car_shared_control).

### Parameter configuraiton

There are a number of important parameters to consider when running the described system.  They are
 * **user_only_control**: 
   * can either be **true** or **false**. 
   * if true, the system operates in user only control.  if false, the system operates in shared control
 * **main_joystick**: 
   * can either be '**right**' or '**left**'
   * this is dependent on the handedness of the human operator
 * **max_time**: 
   * the maximum amount of time for a given trial (in **seconds**)
   * in the paper, this is 20 for the balance bot and 30 for the race car
 * **inflation_radius**: 
   * the amount by which we **inflate the safety barrier**
   * in the paper, this is pi/4 for the balance bot and 3.0 for the race car
 * **time_horizon**: 
   * the **number of discrete time steps** to consider when evaluating the safety of a sampled input
   * in this paper, this is 30 for the balance bot and 25 for the race car
   
### Additional commands
 * **Space bar** : this will shutdown either simulator
 * **'r' key** : this will restart a trial in either simulator
 * **'s' key** : this will start the first trial in the balance bot (this command is not necessary in the race car environment)

### Requirements

**System**
1. **PS3 controller** (or similar input device)
2. **ROS** (tested with ROS Indigo and Ubuntu 14.04)
3. **Python** (tested with Python 2.7)
4. **NVidia GPU** (tested with 2GB NVidia GeForce 860M)
5. **sixad** to connect to PS3 joystick

**Python**
1. **numpy**
2. **pybullet** (simulator for balance bot environment)
3. **pyglet** (for keyboard interaction in race car environment)

**CUDA**
1. **Cuda** (tested with CUDA 7.5)
2. **CuBLAS** 
3. **nvcc** to compile code

Please let us know if you come accross additional requirements when running the system yourself.

### Citing
If you find this code useful in your research, please consider citing:
```Shel
@inproceedings{broad2019highly,
    Author = {Alexander Broad, Todd Murphey and Brenna Argall},
    Title = {Highly Parallelized Data-driven MPC for Minimal Intervention Shared Control},
    Booktitle = {Robotics: Science and Systems (RSS)},
    Year = {2019}
}
  ```

### License

This code is released under the MIT License.