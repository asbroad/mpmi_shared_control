#!/usr/bin/env python

import gym
import time
import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import Polygon, Point32
from race_car_shared_control.msg import RoadPoly
from race_car_shared_control.srv import Mpmi
from polygon import inflate
from pyglet.window import key
import numpy as np

class Simulator():

  def __init__(self):

    # initalize node
    rospy.init_node('simulator')

    # register shutdown hook
    rospy.on_shutdown(self.shutdown_hook)
    self.called_shutdown = False

    # keep track of current keystroke
    self.input_action = [0., 0., 0.]

    # define random seeds for experiment
    self.random_seeds = [13, 65, 99, 200, 56, 3000, 171, 800, 17421, 566]

    # build environment
    self.trial_idx = 0
    self.env = gym.make('CarRacing-v0')
    self.env.seed(self.random_seeds[self.trial_idx])
    self.env.reset()
    self.env.render()

    # set up keystroke hooks
    self.env.unwrapped.viewer.window.on_key_press = self.key_press
    self.terminate = False
    self.restart = False

    # set up subscribers and publishers
    rospy.Subscriber('/joy', Joy, self.joy_callback)
    self.reset_pub = rospy.Publisher('/reset', String, queue_size=1)
    self.shutdown_pub = rospy.Publisher('/shutdown', String, queue_size=1)
    self.road_poly_pub = rospy.Publisher('/road_poly', RoadPoly, queue_size=1)

    # get params
    self.max_time = rospy.get_param('max_time')
    self.inflation_radius = rospy.get_param('/inflation_radius', 0.0)
    self.user_only_control = rospy.get_param('/user_only_control', False)
    self.main_joystick = rospy.get_param('main_joystick', 'right')
    print('User only control : ' + str(self.user_only_control))

    # generate road poly list
    self.generate_road_poly_msg()

    # set up MPPI service call
    rospy.wait_for_service('/mpmi_control')
    mpmi_service = rospy.ServiceProxy('/mpmi_control', Mpmi)

    # run system
    r = rospy.Rate(10)
    self.trial_start_time = time.time()
    while not rospy.is_shutdown():

      if (time.time() - self.trial_start_time) > self.max_time:

        self.reset_pub.publish('reset')
        self.restart = False
        self.trial_idx += 1
        self.trial_idx = 0 if self.trial_idx >= 10 else self.trial_idx
        self.env.seed(self.random_seeds[self.trial_idx])
        self.env.reset()
        self.generate_road_poly_msg()
        self.trial_start_time = time.time()

      if not self.is_safe():

        self.reset_pub.publish('reset')
        self.restart = False
        self.trial_idx += 1
        self.trial_idx = 0 if self.trial_idx >= 10 else self.trial_idx
        self.env.seed(self.random_seeds[self.trial_idx])
        self.env.reset()
        self.generate_road_poly_msg()
        self.trial_start_time = time.time()

      if self.restart:

        self.reset_pub.publish('reset')
        self.restart = False
        self.trial_idx += 1
        self.trial_idx = 0 if self.trial_idx >= 10 else self.trial_idx
        self.env.seed(self.random_seeds[self.trial_idx])
        self.env.reset()
        self.generate_road_poly_msg()
        self.trial_start_time = time.time()

      # get user input
      input_angle, input_gas, input_break = self.input_action

      response = None
      try:
        response = mpmi_service(
                      self.env.env.car.hull.position[0],
                      self.env.env.car.hull.position[1],
                      self.env.env.car.hull.angle,
                      self.env.env.car.hull.linearVelocity[0],
                      self.env.env.car.hull.linearVelocity[1],
                      self.env.env.car.hull.angularVelocity,
                      input_angle,
                      input_gas,
                      input_break
                      )
      except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))

      # take step
      if self.user_only_control:
        obs, rew, done, ite = self.env.step(np.array([input_angle, input_gas, input_break]))
      elif response:
        obs, rew, done, ite = self.env.step(np.array([response.u_1, response.u_2, response.u_3]))

      if self.terminate == True:
        self.shutdown_hook()
        break

      # render and keep time
      self.env.render()
      r.sleep()

  def is_safe(self):
    return ( # 1.0 if wheel is on tile, 0.0 if wheel is on grass
            bool(self.env.env.car.wheels[0].tiles) and
            bool(self.env.env.car.wheels[1].tiles) and
            bool(self.env.env.car.wheels[2].tiles) and
            bool(self.env.env.car.wheels[3].tiles)
            )

  def generate_road_poly_msg(self): # update road poly

    # generate road poly list
    road_poly_list = []
    for pose in self.env.env.road_poly:
      p1_l = [pose[0][0][0], pose[0][0][1]]
      p1_r = [pose[0][1][0], pose[0][1][1]]
      p2_r = [pose[0][2][0], pose[0][2][1]]
      p2_l = [pose[0][3][0], pose[0][3][1]]
      if pose[1] != (1,1,1) and pose[1] != (1,0,0): # get rid of corner markers
        road_poly_list.append([p1_l, p1_r, p2_r, p2_l])

    # inflate road poly
    road_poly_inflated = inflate(road_poly_list, self.inflation_radius)

    # create road poly message
    road_poly_array = []
    for poly in road_poly_inflated:
      p1_l = Point32(poly[0][0], poly[0][1], 0) # z is always 0
      p1_r = Point32(poly[1][0], poly[1][1], 0)
      p2_r = Point32(poly[2][0], poly[2][1], 0)
      p2_l = Point32(poly[3][0], poly[3][1], 0)
      road_poly = Polygon([p1_l, p1_r, p2_r, p2_l, p1_l]) # add first again so that it works with point in poly check
      road_poly_array.append(road_poly)
    road_poly_msg = RoadPoly(road_poly_array)

    # send road poly message
    time.sleep(1.0)
    self.road_poly_pub.publish(road_poly_msg)
    time.sleep(1.0)

  def joy_callback(self, data):
    invert = -1
    input_joy = [0., 0.]
    if self.main_joystick == 'right':
      input_joy = [data.axes[3], invert*data.axes[0]]
    elif self.main_joystick == 'left':
      input_joy = [data.axes[1], invert*data.axes[2]]

    self.input_action = [input_joy[1], 0.0, 0.0] # steer, gas, break
    if input_joy[0] >= 0:
      self.input_action[1] = input_joy[0]
    else:
      self.input_action[2] = -input_joy[0]*0.5

  def key_press(self, k, mod):
    if k == key.SPACE:
      self.terminate = True
    if k == key.R:
      self.restart = True

  def shutdown_hook(self):
    if not self.called_shutdown:
      self.called_shutdown = True
      self.shutdown_pub.publish("shutdown")
      print('Shutting down.')

if __name__=='__main__':
  sim = Simulator()
