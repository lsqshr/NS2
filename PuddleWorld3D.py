import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from euclid import *
from itertools import count

from lib.tf_rl.simulation.karpathy_game import GameObject
from utils import decimal_to_ternary


class Stalker(GameObject):

    def __init__(self, position, speed, angular_speed, obj_type, settings):
        super(Stalker, self).__init__(position, speed, obj_type, settings)

        # Face rotation with Yaw-Pitch-Roll angles
        self.num_actions = 27 * 2
        phi = np.random.random_sample() * 2 * np.pi;
        theta = np.random.random_sample() * 2 * np.pi;
        psi = np.random.random_sample() * 2 * np.pi;
        self.angles = Vector3(phi, theta, psi)
        self.angular_speed = angular_speed; # euclid.Vector3
        self.angular_diff = np.pi / 12 
        self.paddlepower = 0.0


    def _is_collisions(self, p):
        if p.x < 0 or p.x > self.settings['world_size'][0] or\
           p.y < 0 or p.y > self.settings['world_size'][1] or\
           p.z < 0 or p.z > self.settings['world_size'][2]:
           return True
        return False

    def move(self, dt):
        p = self.position + dt * self.speed
        if not self._is_collisions(p):
            self.position = p # Only update when this move does not collide with anything


    def steer(self, action_id): 
        print('action:%d' % action_id)
        # Handle rotation
        rotateid = action_id % self.num_actions
        rotatecode = decimal_to_ternary(rotateid, 3)
        rotatecode = [d-1 for d in rotatecode]
        self.angular_speed = Vector3(rotatecode[0] * self.angular_diff, rotatecode[1] * self.angular_diff, rotatecode[2] * self.angular_diff)

        # Handle leap
        if action_id > 26:
            self.paddlepower = 0.05
        else:
            self.paddlepower = 0.0


    def rotate(self, dt):
        self.angles += dt * self.angular_speed
        q1 = Quaternion.new_rotate_axis(self.angles.x, Vector3(1, 0, 0))
        q2 = Quaternion.new_rotate_axis(self.angles.y, Vector3(0, 1, 0))
        q3 = Quaternion.new_rotate_axis(self.angles.z, Vector3(0, 0, 1))
        R = q1 * q2 * q3

        # basevec = np.array([self.paddlepower, 0.0, 0.0, 1.0]);
        basevec = Vector3(self.paddlepower, 0, 0)
        rotatedvec = R * basevec
        print(basevec, rotatedvec)
        self.speed = rotatedvec


    def step(self, dt):
        # self.check_collisions()
        self.rotate(dt)
        self.move(dt)
        print('step with speed %f, %f, %f\tangles %f, %f, %f\tpaddle-power:%f' \
            % (self.speed.x, self.speed.y, self.speed.z, self.angles.x, self.angles.y, self.angles.z, self.paddlepower))

    def draw(self):
        pass


class PuddleWorld3D(object):

    def __init__(self, settings):
        self.settings = settings
        self.size = settings['world_size']
        self.stalker = Stalker(Point3(0.3, 0.3, 0.3), Vector3(0, 0, 0), Vector3(0, 0, 0), "stalker", {'object_radius': 0.01, 'world_size': settings['world_size']}) 


    def perform_action(self, action_id):
        assert(0 <= action_id < self.stalker.num_actions)
        self.stalker.steer(action_id)


    def spawn_object(self, obj_type):
        pass


    def step(self, dt):
        self.stalker.step(dt)


    def resolve_collisions(self):
        pass


    def observe(self):
        pass


    def collect_reward(self):
        pass


    def draw(self):
        pass

# Tests
def teststalker():
    chunks_per_frame = 1
    fps = 30.0
    chunk_length_s = 1.0 / fps
    world = PuddleWorld3D({'world_size': [1, 1, 1]})
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frameidx in count():
        for _ in range(chunks_per_frame):
            world.step(chunks_per_frame)

        new_ob = world.observe()
        reward = world.collect_reward()
        world.perform_action(np.random.randint(27 * 2 - 1))

        # Draw world 
        plt.cla()
        p = world.stalker.position
        ax.scatter(p.x, p.y, p.z)
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
        plt.draw()


if __name__ == '__main__':
    teststalker()
