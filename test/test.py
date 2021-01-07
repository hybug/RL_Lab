'''
Author: hanyu
Date: 2021-01-07 09:46:18
LastEditTime: 2021-01-07 09:49:15
LastEditors: hanyu
Description: test
FilePath: /test_ppo/test/test.py
'''

def wrapClass(cls):
    def inner(a):
        print('class name:', cls.__name__)
        return cls(a)
    return inner

@wrapClass
class Foo():
    def __init__(self, a):
        self.a = a

    def fun(self):
        print('self.a =', self.a)


m = Foo('hello')
m = wrapClass(Foo('hello'))
# m.fun()
