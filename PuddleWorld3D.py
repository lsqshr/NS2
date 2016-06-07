import numpy as np
from euclid import *
from lib.tf_rl.simulation.karpathy_game import GameObject

class Stalker(GameObject):
	def __init__(self, position, speed, obj_type, settings):
		super(Stalker, self).__init__(position, speed, obj_type, settings)

		# Face rotation with Yaw-Pitch-Roll angles
		self.phi = np.random.random_sample() * 2 * np.pi;
		self.theta = np.random.random_sample() * 2 * np.pi;
		self.psi = np.random.random_sample() * 2 * np.pi;


	def check_collisions(self):
		pass


	def move(self, dt):
		pass


	def step(self, dt):
		pass


# Tests
def teststalker():
	s = Stalker(Point3(0.3, 0.3, 0.3), 0.01, "stalker", {'object_radius': 0.01}) 


if __name__ == '__main__':
	teststalker()
