#include<iostream>
using namespace std;

class Base {
public:
	virtual void virFunc() {
		cout << "Base virtual" << endl;
	}
	virtual void virFunc1() {
		cout << "Base virtual-1" << endl;
	}
};

class Derived :public Base {
public:
	void virFunc() {
		cout << "Derived virtual" << endl;
	}
};

class Derived1 :public Derived {
public:
	void virFunc() {
		cout << "__________________" << endl;
	}
};


void main() {
	Base* base = new Base();
	base->virFunc();
	base->virFunc1();
	Base* bs1 = new Derived();
	bs1->virFunc();
	bs1->virFunc1();
	Base* bs2= new Derived1();
	bs2->virFunc();
	bs2->virFunc1();
}