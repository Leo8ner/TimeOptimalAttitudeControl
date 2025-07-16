#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#
# -*- coding: utf-8 -*-
from casadi import *
#
# How to use Callback
# Joel Andersson
#

class MyCallback(Callback):
  def __init__(self, name, d, opts={}):
    Callback.__init__(self)
    self.d = d
    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, arg):
    x = arg[0]
    f = sin(self.d*x)
    return [f]

# Use the function
f = MyCallback('f', 0.5)
res = f(2)
print(res)

# You may call the Callback symbolically
x = MX.sym("x")
print(f(x))

# Derivates OPTION 1: finite-differences
eps = 1e-5
print((f(2+eps)-f(2))/eps)

f = MyCallback('f', 0.5, {"enable_fd":True})
J = Function('J',[x],[jacobian(f(x),x)])
print(J(2))

class Example4To3(Callback):
  def __init__(self, name, opts={}):
    Callback.__init__(self)
    self.construct(name, opts)

  def get_n_in(self): return 1
  def get_n_out(self): return 1

  def get_sparsity_in(self,i):
    return Sparsity.dense(4,1)

  def get_sparsity_out(self,i):
    return Sparsity.dense(3,1)

  # Evaluate numerically
  def eval(self, arg):
    a,b,c,d = vertsplit(arg[0])
    ret = vertcat(sin(c)*d+d**2,2*a+c,b**2+5*c)
    return [ret]

# Derivates OPTION 4: Supply full Jacobian

class Example4To3_Jac(Example4To3):
  def has_jacobian(self): return True
  def get_jacobian(self,name,inames,onames,opts):
    class JacFun(Callback):
      def __init__(self, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

      def get_n_in(self): return 2
      def get_n_out(self): return 1

      def get_sparsity_in(self,i):
        if i==0: # nominal input
          return Sparsity.dense(4,1)
        elif i==1: # nominal output
          return Sparsity(3,1)

      def get_sparsity_out(self,i):
        return sparsify(DM([[0,0,1,1],[1,0,1,0],[0,1,1,0]])).sparsity()

      # Evaluate numerically
      def eval(self, arg):
        a,b,c,d = vertsplit(arg[0])
        ret = DM(3,4)
        ret[0,2] = d*cos(c)
        ret[0,3] = sin(c)+2*d
        ret[1,0] = 2
        ret[1,2] = 1
        ret[2,1] = 2*b
        ret[2,2] = 5
        return [ret]

    # You are required to keep a reference alive to the returned Callback object
    self.jac_callback = JacFun()
    return self.jac_callback

f = Example4To3_Jac('f')
x = MX.sym("x",4)
J = Function('J',[x],[jacobian(f(x),x)])
print(J(vertcat(1,2,0,3)))

    