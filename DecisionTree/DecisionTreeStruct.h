#include <iostream>  
#include <string>  
#include <vector>  
#include <map>  
#include <algorithm>  
#include <cmath>  
using namespace std;


struct Node {//�������ڵ�  
	string attribute;//����ֵ  
	string arrived_value;//���������ֵ  
	vector<Node *> childs;//���еĺ���  
	Node() {
		attribute = "";
		arrived_value = "";
	}
};
Node * root; 
