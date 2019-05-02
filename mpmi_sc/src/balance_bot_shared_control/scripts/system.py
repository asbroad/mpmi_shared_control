#!/usr/bin/env python

import gym
import time
import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import String, Float32
from balance_bot_shared_control.srv import Mpmi
import pybullet as p
import numpy as np

class Simulator():

  def __init__(self):

    # initalize node
    rospy.init_node('simulator')
    self.user_action = 0.0
    self.max_angle = 1.768

    # register shutdown hook
    rospy.on_shutdown(self.shutdown_hook)
    self.called_shutdown = False

    # set up subscriber
    rospy.Subscriber('/joy', Joy, self.joy_callback)
    self.reset_pub = rospy.Publisher('/reset', String, queue_size=1)
    self.shutdown_pub = rospy.Publisher('/shutdown', String, queue_size=1)
    self.inflation_pub = rospy.Publisher('/inflation_update', Float32, queue_size=1)

    # get params
    self.max_time = rospy.get_param('max_time')
    self.inflation_radius = rospy.get_param('/inflation_radius', np.pi/2.0)
    self.user_only_control = rospy.get_param('/user_only_control', False)
    self.main_joystick = rospy.get_param('main_joystick', 'right')
    print('User only control : ' + str(self.user_only_control))

    # set up MPPI service call
    rospy.wait_for_service('/mpmi_control')
    mpmi_service = rospy.ServiceProxy('/mpmi_control', Mpmi)

    # build environment
    self.env = gym.make('balancebot-v0')

    # press 's' key to start the trials
    pos_or_neg = 1.0
    start = False
    while not start:
      keys = p.getKeyboardEvents()
      for key in keys:
        if str(key) == '115': # s key starts trial
          self.env.reset()
          pos_or_neg = 1.0 if np.random.random() < 0.5 else -1.0
          obs, rew, done, ite = self.env.step(pos_or_neg*np.random.normal(0.5, 1.0))
          self.env.render('human')
          start = True

    # set up hooks
    self.terminate = False
    self.restart = False

    self.trial_start_time = time.time()
    r = rospy.Rate(50)

    # run system with input from user
    while not rospy.is_shutdown():

      # get keyboard input
      keys = p.getKeyboardEvents()
      for tup in keys.items():
        key = tup[0]
        val = tup[1]
        if key == 32: # space bar
          self.terminate = True
        elif key == 114 and val == 4: # lower case 'r' key
          self.restart = True

      if (time.time() - self.trial_start_time) > self.max_time:
        self.restart = False
        self.env.reset()
        pos_or_neg = 1.0 if np.random.random() < 0.5 else -1.0
        obs, rew, done, ite = self.env.step(pos_or_neg*np.random.normal(0.5, 1.0))
        self.reset_pub.publish('reset')
        time.sleep(0.2)
        self.trial_start_time = time.time()

      if not self.is_safe():
        self.restart = False
        self.env.reset()
        pos_or_neg = 1.0 if np.random.random() < 0.5 else -1.0
        obs, rew, done, ite = self.env.step(pos_or_neg*np.random.normal(0.5, 1.0))
        self.reset_pub.publish('reset')
        time.sleep(0.2)
        self.trial_start_time = time.time()

      if self.restart:
        self.restart = False
        self.env.reset()
        pos_or_neg = 1.0 if np.random.random() < 0.5 else -1.0
        obs, rew, done, ite = self.env.step(pos_or_neg*np.random.normal(0.5, 1.0))
        self.reset_pub.publish('reset')
        time.sleep(0.2)
        self.trial_start_time = time.time()

      # get user input
      control_input = self.user_action

      # get current state
      cur_angle, cur_angle_dot, cur_velocity = self.env._compute_observation()

      response = None
      try:
        response = mpmi_service(
                      cur_angle,
                      cur_angle_dot,
                      cur_velocity,
                      control_input
                      )
      except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))

      # take step
      if self.user_only_control:
        obs, rew, done, ite = self.env.step(control_input)
      elif response:
        obs, rew, done, ite = self.env.step(response.u_1)
      else:
        obs, rew, done, ite = self.env.step(control_input)

      if self.terminate == True:
        self.shutdown_hook()
        break

      self.env.render('human')
      r.sleep()

  def is_safe(self):
    cubePos, cubeOrn = p.getBasePositionAndOrientation(self.env.botId)
    cubeEuler = p.getEulerFromQuaternion(cubeOrn)
    return cubeEuler[0] < self.max_angle and cubeEuler[0] > -self.max_angle

  def joy_callback(self, data):
    if self.main_joystick == 'right':
      self.user_action = data.axes[2]
    elif self.main_joystick == 'left':
      self.user_action = data.axes[0]

  def shutdown_hook(self):
    if not self.called_shutdown:
      self.called_shutdown = True
      self.shutdown_pub.publish("shutdown")
      print('Shutting down.')

if __name__=='__main__':
  sim = Simulator()
