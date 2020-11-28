try:
    from point import Point
except ImportError as e:
    print(e)

p1 = Point(10, 20)
print(p1)
print(p1.pri())

p2 = Point()
print(p2)

p2.set_x(100)
print(p2)

