class Account:
    num_accounts = 0  # 클래스 변수 : 이 클래스를 기반으로 만들어진 인스턴스들이 모두 공유하는 변수.

    def __init__(self, name):
        # __init__ : 생성자라 불리며 initialize(초기화하다)의 약자. 인스턴스가 생성될 때 자동으로 한번만 실행되는 메소드.
        # self : 첫 번째 매개변수는 관례적으로 self를 사용한다. self는 클래스 인스턴스(a, b, c)를 가르킨다.
        # name : 매개변수로 함수와 메소드를 정의시 사용한다.
        self.his_name = name
        # his_name : 인스턴스 변수로 각각의 인스턴스마다 독립적인 값을 갖는 변수이다.

        Account.num_accounts += 1

    # 클래스 변수를 사용할 때에는 '클래스명.클래스 변수명'의 형태로 사용한다.

    def __del__(self):
        Account.num_accounts -= 1


# self가 필요한 이유 : 클래스를 정의할 때 self가 없다면 'self.his_name = name' 구문에서 '(    ).his_name = name'이
# 되는데 위에서부터 순차적으로 소스를 읽어나갈 때는 아래의 a = Account('KIM')과 같이 인스턴스의 이름을 알 수 없다.
# 그래서 이를 방지하기 위해 이 클래스에 어떤 클래스 인스턴스가 들어올 때 self로 받게 된다.

a = Account('KIM')
# 'KIM', 'HONG', 'LEE'는 인수로 매개변수에 실제로 담는 값을 의미한다.

b = Account('HONG')
# b는 name = 'HONG'이 들어간 Account class 형태의 인스턴스(객체)를 가르키고 있다.(참조)
# 이는 추상적인 클래스를 실제적인 객체로 만든 것으로 인스턴스화 하는 것이다.
# 여기서 인스턴스란 객체 지향 프로그래밍의 특징으로 할당된 메모리 공간 자체를 의미한다.
# 이는 클래스 인스턴스를 생성한 것과 거의 같으며 a, b, c는 클래스 인스턴스라 할 수 있다.

c = Account('LEE')

a.__del__()  # self는 인수를 받는 매개 변수가 아니므로 공란으로 두어도 된다. Account.num_accounts는 1이 줄어든다.
b.__init__('CHOI')  # 초기화 과정인 __init__을 다시 진행해준다. 기존의 b.his_name = 'HONG'에서 'CHOI'로 바뀌게 된다.

print(Account.num_accounts)  # 3 / print(a.num_accounts) = print(b.num_accounts)
print(a.num_accounts)
print(b.num_accounts)
print(a.his_name)  # 인스턴스 변수는 클래스 외부에서 이렇게 사용된다.
print(b.his_name)


