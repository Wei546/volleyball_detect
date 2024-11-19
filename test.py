class Test:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def plus(self,num1,num2):
        result = self.x + self.y + (num1*num2)
        return result
    def minus(self):
        return self.x - self.y
    def getNum(self):
        test = Test(self.x,self.y)
        print(f'x: {test.x}, y: {test.y}')
test = Test(1,2)
print(test.plus())
