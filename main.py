import numpy as np
from matplotlib import pyplot as pyplt
import matplotlib as plt
import matplotlib.animation as animation

#定义粒子
class Particles:
    #初始化粒子属性
    def __init__(self, coord, mass=1, velocity=0, acceleration=0, **properties):
        self.coord = np.array(coord,dtype='float64')
        self.mass = mass
        if velocity:
            self.velocity = np.array(velocity,dtype='float64')
        else:
            self.velocity = np.zeros(len(coord))
        if acceleration:
            self.acceleration = np.array(acceleration,dtype='float64')
        else:
            self.acceleration = np.zeros(len(coord))
        for prop in properties:
            setattr(self, prop, properties[prop])
        self.__params__ = ["coord", "mass", "velocity", "acceleration"] + list(properties.keys())
    #定义速度作用
    def move(self, dt):
        self.coord += self.velocity*dt
    #定义加速度作用
    def accelerate(self, dt):
        self.velocity += self.acceleration*dt
    #定义外力作用
    def force(self, force):
        self.acceleration += force / self.mass
    #定义惯性
    def no_force(self):
        self.acceleration *= 0
    #输出粒子属性详情
    def detail(self):
        for prop in self.__params__:
            print("{}: {}".format(prop, getattr(self, prop)))
    def __repr__(self):
        return str(self.coord)

#定义场
class Field:
    #初始化场相互作用、作用因子、维数
    def __init__(self, interaction, factor, dimensions):
        self.interaction = interaction#相互作用力应由一关于（作用粒子，被作用粒子，粒子间距）的函数给出
        self.factor = factor
        self.dimensions = dimensions
        self.particles = []
        self.__params__ = ["interaction", "factor", "dimensions", "particles"]
    #定义粒子添加
    def append(self, *particles):
        for particle in particles:
            self.particles.append(particle)
    #定义作用强度计算
    def intensity(self, coord):
        single_point = Particles(np.zeros(self.dimensions))
        setattr(single_point, self.factor, 1)
        #print(single_point.detail())
        resultant_force = np.zeros(self.dimensions)
        for particle in self.particles:
            d = np.linalg.norm(coord - particle.coord)
            if d == 0:
                continue
            fmod = self.interaction(getattr(particle, self.factor), getattr(single_point, self.factor), d)
            resultant_force += fmod * (particle.coord - coord) / d
        return resultant_force
    #定义步进仿真
    def step(self, dt=0.01):
        for particle in self.particles:
            particle.no_force()
            particle.force(self.intensity(particle.coord)*getattr(particle, self.factor))
            particle.accelerate(dt)
            particle.move(dt)
            #particle.detail()
    #清空场内粒子
    def clean(self):
        self.particles = []
    #输出作用场属性详情
    def detail(self):
        for prop in self.__params__:
            if isinstance(getattr(self, prop), list):
                print ("Amount of {}: {}".format(prop, len(getattr(self, prop))))
            print("{}: {}\n".format(prop, getattr(self, prop)))

#定义仿真环境
class Space:
    def __init__(self):
        self.particles = []
        self.fields = []
    def append(self, *args, **kwargs):
        self.particles.append(Particles(*args, **kwargs))
    def field(self,*args):
        self.fields.append(*[field for field in args])
    def step(self, dt=0.01):
        for field in self.fields:
            field.append(*self.particles)
            #print("{} is interacting:\n{}".format(field.factor,field.particles))
            field.step(dt)
            field.clean()
    def render(self, t, dt=0.01):
        i = 0.
        self.state = [[list(particle.coord) for particle in self.particles]]
        while i < t:
            self.step(dt)
            state_next = [list(particle.coord) for particle in self.particles]
            #print('\n\n\nstate_next = {}\n\nstate = {}\n\n\n'.format(state_next,self.state))
            self.state.append(state_next)
            i += dt
    def plot(self, t, dt=0.01):
        X = []
        Y = []
        self.render(t, dt)
        for i in range(len(self.state)):
            x, y = zip(*self.state[i])
            #print('\n\nx={}\n************************************'.format(x))
            X += x
            Y += y
            #print('after:\nX={}\n###################################'.format(X))
        
        pyplt.figure(figsize=[8, 8])
        pyplt.scatter(X, Y)
        pyplt.scatter(*zip(*self.state[0]), c="red")
        pyplt.scatter(*zip(*self.state[-1]), c="orange")
        pyplt.show()


s = Space()

for i in range(3): s.append(10*np.random.randn(2),mass = np.random.rand()*10**6, q = (np.random.rand())*10**(-4))
#s.append([0,0],mass = 1,q = 1)

e = Field(lambda q2, q1, r: -9*(10**9)*q2*q1/(r**2), 'q', 2)

g = Field(lambda m2, m1, r: 6.67*10**(-11)*m2*m1/(r**2), 'mass', 2)

#s.field(e)

s.field(g)

print(s.particles)

print(s.fields)

s.plot(t=100,dt=0.01)

print("test start!")
