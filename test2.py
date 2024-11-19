from test import Test
class Test2(Test):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a = a
        self.b = b
    def multiply(self):
        test1 = Test.plus(self)
        return self.a * self.b + test1
test2 = Test2(1,2)
print(test2.multiply())