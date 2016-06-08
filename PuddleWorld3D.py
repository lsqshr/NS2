import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from euclid import *
from itertools import count

from lib.tf_rl.simulation.karpathy_game import GameObject
from utils import decimal_to_ternary


class Wall(object):
    
    def __init__(self, center, rotatevec, width, height):
        self.center = center
        self.width = width
        self.height = height
        self.rotatevec = rotatevec
        self._genpts()


    def _genpts(self):
        self._pts = []
        basecenter = Point3(0, 0, 0)
        basepoints = []
        X = np.arange(-self.width/2, self.width/2, 0.02)
        Y = np.arange(-self.height/2, self.height/2, 0.02)

        # Get rotation matrix
        q1 = Quaternion.new_rotate_axis(self.rotatevec.x, Vector3(1, 0, 0))
        q2 = Quaternion.new_rotate_axis(self.rotatevec.y, Vector3(0, 1, 0))
        q3 = Quaternion.new_rotate_axis(self.rotatevec.z, Vector3(0, 0, 1))
        R = q1 * q2 * q3

        # Get mesh
        pts = [R * Point3(x, y, 0) for x, y in zip(X, Y)]
        X, Y = np.meshgrid([p.x + self.center.x for p in pts], [p.y + self.center.y for p in pts])
        self.meshx = X
        self.meshy = Y
        self.meshz = [p.z + self.center.z for p in pts]

    
    def check_collision(self, oldp, newp):
        # It now assumes the wall is orthogonal to one axis
        if self.rotatevec.x > 0: # Find the axis it is orthogonal to
            t = self.center.y
            if (oldp.y - t) * (newp.y - t) < 0:
                return True
            else:
                return False
        elif self.rotatevec.y > 0:
            t = self.center.x
            if (oldp.x - t) * (newp.x - t) < 0:
                return True
            else:
                return False
        else: # It means it is still parallel to Z plane, only the z axis matters now
            t = self.center.z
            if (oldp.z - t) * (newp.z - t) < 0:
                return True
            else:
                return False
 

    def draw(self, ax):
        ax.plot_wireframe(self.meshx, self.meshy, self.meshz)


class Stalker(GameObject):

    def __init__(self, position, speed, angular_speed, walls, settings):
        super(Stalker, self).__init__(position, speed, 'stalker', settings)

        # Face rotation with Yaw-Pitch-Roll angles
        self.num_actions = 27 * 2
        phi = np.random.random_sample() * 2 * np.pi;
        theta = np.random.random_sample() * 2 * np.pi;
        psi = np.random.random_sample() * 2 * np.pi;
        self.angles = Vector3(phi, theta, psi)
        self.angular_speed = angular_speed; # euclid.Vector3
        self.angular_diff = np.pi / 12 
        self.paddlepower = 0.0
        self.walls = walls


    def _is_collisions(self, p):
        # Check the world boundary
        if p.x < 0 or p.x > self.settings['world_size'][0] or\
           p.y < 0 or p.y > self.settings['world_size'][1] or\
           p.z < 0 or p.z > self.settings['world_size'][2]:
           return True
        
        for w in self.walls:
            if w.check_collision(self.position, p):
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
            self.paddlepower = self.settings['paddlepower']
        else:
            self.paddlepower = 0.0


    def rotate(self, dt):
        self.angles += dt * self.angular_speed
        q1 = Quaternion.new_rotate_axis(self.angles.x, Vector3(1, 0, 0))
        q2 = Quaternion.new_rotate_axis(self.angles.y, Vector3(0, 1, 0))
        q3 = Quaternion.new_rotate_axis(self.angles.z, Vector3(0, 0, 1))
        R = q1 * q2 * q3

        basevec = Vector3(self.paddlepower, 0, 0)
        rotatedvec = R * basevec
        print(basevec, rotatedvec)
        self.speed = rotatedvec


    def step(self, dt):
        # self.check_collisions()
        self.rotate(dt)
        self.move(dt)
        print('step with speed %f, %f, %f\tangles %f, %f, %f\tpaddle-power:%f\tX:%f, Y:%f, Z:%f' \
            % (self.speed.x, self.speed.y, self.speed.z, self.angles.x, self.angles.y, self.angles.z,\
               self.paddlepower, self.position.x, self.position.y, self.position.z))

    def draw(self, ax):
        p = self.position
        s = self.speed * 2
        ax.scatter(p.x, p.y, p.z)
        ax.plot([p.x, p.x + s.x], [p.y, p.y + s.y],[p.z, p.z + s.z])


class PuddleWorld3D(object):

    def __init__(self, settings):
        self.settings = settings
        self.size = settings['world_size']
        self.walls = []
        self.walls.append(Wall(Point3(0.7, 0.5, 0.5), Vector3(0, np.pi/2, 0), 0.6, 0.6))
        stalkersettings = {'object_radius': 0.01, \
                           'world_size': settings['world_size'], \
                           'paddlepower': settings['paddlepower']}
        self.stalker = Stalker(Point3(0.5, 0.5, 0.5), Vector3(0, 0, 0), Vector3(0, 0, 0), self.walls, stalkersettings) 
        self.observation_size = 6 # Without coarse coding


    def perform_action(self, action_id):
        assert(0 <= action_id < self.stalker.num_actions)
        self.stalker.steer(action_id)


    def spawn_object(self, obj_type):
        self.stalker = Stalker(Point3(0.3, 0.3, 0.3), Vector3(0, 0, 0), Vector3(0, 0, 0),\
                 "stalker", {'object_radius': 0.01, 'world_size': settings['world_size'], 'paddlepower': settings['paddlepower']}) 


    def step(self, dt):
        self.stalker.step(dt)


    def observe(self):
        observation = np.zeros(self.observation_size)
        p = self.stalker.position
        s = self.stalker.speed
        observation[0:3] = [p.x, p.y, p.z]
        observation[3:]  = [s.x, s.y, s.z]
        return observation


    def collect_reward(self):



    def draw(self, ax):
        for w in self.walls:
            w.draw(ax)
        self.stalker.draw(ax)
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)

# Tests
def testdrawwall():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    wall = Wall(Point3(0.7, 0.7, 0.8), Vector3(0, np.pi/2, 0), 0.6, 0.6)
    wall.draw(ax)
    plt.show()


def teststalker():
    
    # Randomly Generate Action ID for stalker
    chunks_per_frame = 1
    fps = 30.0
    chunk_length_s = 1.0 / fps
    settings = {'world_size': [1, 1, 1], 'paddlepower': 0.01}
    world = PuddleWorld3D(settings)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frameidx in count():
        for _ in range(chunks_per_frame):
            world.step(chunks_per_frame)

        new_ob = world.observe()
        reward = world.collect_reward()
        world.perform_action(np.random.randint(27 * 2 - 1))
        plt.cla()
        world.draw(ax) # Draw world
        plt.draw()


if __name__ == '__main__':
    teststalker()
