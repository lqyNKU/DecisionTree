#include "DecisionTree.h"
#include<iostream>
#include<windows.h>
#include<stdlib.h>
#include <pmmintrin.h>

using namespace std;

//num sex married graduated high_salary label
//1 0 0 1 1 yes
//2 1 0 1 0 yes
//3 0 1 0 1 no
//end
//num sex married graduated high_salary label
//1 0 0 1 1 yes
//2 1 0 1 0 yes
//3 0 1 1 1 no
//4 1 0 0 0 no
//5 1 1 0 0 yes
//6 0 1 0 0 no
//7 0 0 0 0 no
//8 0 0 1 0 no
//9 1 1 0 1 yes
//10 1 1 1 1 yes
//11 0 1 0 0 no
//12 0 0 1 1 no
//13 0 1 0 1 yes
//14 1 0 0 1 no
//15 1 0 1 1 yes
//16 0 1 1 0 no
//end


vector<vector<string>> DecisionTree::GetState() {
	return this->state;
}

vector<string> DecisionTree:: GetAttribute() {
	return this->attribute_row;
}

//������
DecisionTree::DecisionTree(vector<vector<string>> state, int tree_size) {
	this->state = state;
	this->tree_size = tree_size;
}

//��������ʵ������������ֵ��ɵ�map  
void DecisionTree::ComputeMapFrom2DVector() {
	//cout << "The size of map_attribute_values is: " << map_attribute_values.size() << endl;
	
	unsigned int i, j, k;
	bool exited = false;
	vector<string> values;//�洢�������µ����в��ظ��Ŀ���ȡֵ
	for (i = 1; i < MAXLEN - 1; i++) {//�����б���  
		for (j = 1; j < state.size(); j++) {
			for (k = 0; k < values.size(); k++) {
				//string�ıȽϣ����Ϊ0�������Ϊ-1
				//��ֹ�ظ�ֵ����
				if (!values[k].compare(state[j][i])) exited = true;
			}
			if (!exited) {
				values.push_back(state[j][i]);//ע��Vector�Ĳ��붼�Ǵ�ǰ�����ģ�ע�����it��ʼ��ָ��vectorͷ  
			}
			exited = false;
		}
		map_attribute_values[state[0][i]] = values;//������state[0][i]��values��Ӧ�γ�һ��map
		values.erase(values.begin(), values.end());
	}
}

//�������ڴ���ComputeGini
vector<double> DecisionTree::ComputeEntropy(vector<vector<int>> numOfEachAttr, vector<int> proOfEachAttr) {
	vector<double> entropy_sum;
	for (int i = 0; i < numOfEachAttr.size(); i++) {
		int sum1 = 0;
		int sum3 = 0;
		int pro = proOfEachAttr[i];
		double p1, p3;
		sum1 = numOfEachAttr[i][0] + numOfEachAttr[i][1];
		sum3 = numOfEachAttr[i][2] + numOfEachAttr[i][3];
		//cout << "sum1 " << sum1 << "  " << sum3 << endl;
		p1 = (double)numOfEachAttr[i][0] / (double)sum1;
		p3 = (double)numOfEachAttr[i][2] / (double)sum3;
		double entropy = (double)pro / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
		entropy += (1 - (double) pro / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));
		entropy_sum.push_back(entropy);
	}
	return entropy_sum;
	
	
}


//���а棺���ݾ������Ժ�ֵ������ĳ����ֵ�»���ֵ
vector<double> DecisionTree::ComputeEntropyParallel(vector<vector<int>> numOfEachAttr, vector<int> proOfEachAttr) {
	vector<double> entropy_sum;
	for (int i = 0; i < numOfEachAttr.size(); i++) {
		
		float x, y, z, w;
		x = numOfEachAttr[i][0];
		y = numOfEachAttr[i][2];
		z = numOfEachAttr[i][1];
		w = numOfEachAttr[i][3];
		
		float* op1 = new float[2];
		float* op2 = new float[2];
		op1[0] = x;
		op1[1] = y;
		op2[0] = z;
		op2[1] = w;
		
		__m128 a, b, c, d, e, f;
		a = _mm_loadu_ps(op1);
		b = _mm_loadu_ps(op2);
	
		c = _mm_add_ps(a, b);//sum1,sum3
		d = _mm_div_ps(a, c);//p1,p3
		c = _mm_hadd_ps(c, c);//sum1+sum3,sum1+sum3
		
		float* pro = new float[2];
		pro[0] = proOfEachAttr[i];
		pro[1] = 1 - proOfEachAttr[i];
		e = _mm_loadu_ps(pro);//pro,1-pro

		f = _mm_div_ps(e, c);//pro/sum
		
		a = _mm_setzero_ps();
		b = _mm_setzero_ps();
		float* op3 = new float[2];
		op3[0] = 1;
		op3[1] = 1;
		a = _mm_loadu_ps(op3);//1,1
		b = _mm_sub_ps(a, d);//1-p1,1-p3

		b = _mm_mul_ps(b, b);
		d = _mm_mul_ps(d, d);//p1*p1,p3*p3
		a = _mm_sub_ps(a, b);//1-p1*p1,1-p3*p3
		a = _mm_sub_ps(a, d);
		f = _mm_mul_ps(f, a);

		float res = 0;
		
		_mm_storeu_ps(&res, f);
		double entropy = (double)res;
		
		entropy_sum.push_back(entropy);
	}
	return entropy_sum;
}
//double DecisionTree::ComputeGini_Parallel(vector <vector <string> > remain_state,
//	string attribute, string value, bool ifparent) {
//	vector<int> count(4, 0);//��ʼ��count����������0
//	unsigned int i, j;
//	bool done_flag = false;
//	int proportion;
//	for (j = 1; j < MAXLEN; j++) {
//		if (done_flag) break;
//		//�ҵ������������
//		if (!attribute_row[j].compare(attribute)) {
//			proportion = 0;
//			for (i = 1; i < remain_state.size(); i++) {
//				//���˴�����ֵΪvalue
//				if (!remain_state[i][j].compare(value)) {
//					proportion++;
//					//yes��no�ֱ�Ϊ����label
//					if (!remain_state[i][MAXLEN - 1].compare(yes)) {
//						count[0] ++;
//					}
//					else count[1] ++;
//				}
//				//�˴�����ֵ��Ϊvalue
//				else {
//					//yes��no�ֱ�Ϊ����label
//					if (!remain_state[i][MAXLEN - 1].compare(yes)) {
//						count[2] ++;
//					}
//					else count[3] ++;
//				}
//			}
//			done_flag = true;
//		}
//	}
//
//	//// ȫ����������ȫ�Ƿ���
//	//if (count[0] == 0 || count[1] == 0)  return 0;
//
//	/****************************************************************************************/
//	//����gini
//	int sum = count[0] + count[1];
//	double p1 = count[0] * 1.0 / sum;
//	double p2 = count[1] * 1.0 / sum;
//	double entropy = (1 - p1 * p1 - p2 * p2)*(double)proportion / (double)remain_state.size();
//	/****************************************************************************************/
//	return entropy;
//}


//���ݾ������Ժ�ֵ������ĳ����ֵ�»���ֵ
vector<double> DecisionTree::ComputeGini(vector<vector <string>> remain_state,
	vector<string> remain_attribute) {
	

	vector<vector<int>> count_sum;//ÿһ�����������ĸ�ֵ
	
	unsigned int i, j;
	bool done_flag = false;
	vector<int>proportion_sum;//ÿ�����Զ�Ӧһ��
	
	vector<string>::iterator it = remain_attribute.begin();

	for (j = 1; j < MAXLEN; j++) {
		//ÿ�ε�remain_attribute������ͷ�ļ����к�ĩβ�ı�ǩ�У���ȥ��
		for (it = remain_attribute.begin() + 1; it < remain_attribute.end() - 1; it++) {
			//�ҵ���Ӧ����
			if (!attribute_row[j].compare(*it)) {
				int proportion = 0;	
				vector<string>::iterator it_value = map_attribute_values[*it].begin();
				vector<int> count(4, 0);//��ʼ��count�������ĸ�0��ǰ���ͺ���ֱ��Ӧһ������
				//cout << count[0] << endl;
				for (i = 1; i < remain_state.size(); i++) {
					//���˴�����ֵΪvalue
					if (!remain_state[i][j].compare(*it_value)) {
						proportion++;
						//yes��no�ֱ�Ϊ����label
						if (!remain_state[i][MAXLEN - 1].compare(yes)) {
							count[0] ++;
						}
						else count[1] ++;
					}
					//�����ֻ࣬�����ֿ���
					else {
						if (!remain_state[i][MAXLEN - 1].compare(yes)) {
							count[2] ++;
						}
						else count[3] ++;
					}
				}
				count_sum.push_back(count);
				proportion_sum.push_back(proportion);
			}
		}
	}
	//cout << "count " << count_sum[1][2] << endl;
	//cout << "pro " << proportion_sum[2] << endl;
	vector<double> entropy(ComputeEntropy(count_sum, proportion_sum));
	//vector<double> entropy(ComputeEntropyParallel(count_sum, proportion_sum));
	return entropy;
	
}

// ���㰴������attribute���ֵ�ǰʣ��ʵ������Ϣ����
//double DecisionTree::ComputeGain(vector <vector <string> > remain_state, string attribute) {
//	unsigned int j, k, m;
//	double parent_entropy = ComputeGini(remain_state, attribute, blank, true);
//	double children_entropy = 0;
//
//	vector<string> values = map_attribute_values[attribute];//ȡ���������µĸ���ֵ
//	vector<double> ratio;
//	vector<int> count_values;
//	int tempint;
//	for (m = 0; m < values.size(); m++) {
//		tempint = 0;
//		for (k = 1; k < MAXLEN - 1; k++) {
//			if (!attribute_row[k].compare(attribute)) {
//				for (j = 1; j < remain_state.size(); j++) {
//					//����ǰ��ĳʣ��ʵ��������ֵ��values��Ӧ�����tempint+1
//					if (!remain_state[j][k].compare(values[m])) {
//						tempint++;
//					}
//				}
//			}
//		}
//		count_values.push_back(tempint);//��ÿһvalues��ȵ�ʵ���ĸ���
//	}
//
//	for (j = 0; j < values.size(); j++) {
//		//remain_state��һ��Ϊ���Ա�ʶ
//		//ratioΪ��values��ĳֵ��ȵı���
//		ratio.push_back((double)count_values[j] / (double)(remain_state.size() - 1));
//	}
//	double temp_entropy;
//	for (j = 0; j < values.size(); j++) {
//		temp_entropy = ComputeGini(remain_state, attribute, values[j], false);
//		children_entropy += ratio[j] * temp_entropy;
//	}
//	return (parent_entropy - children_entropy);
//}

int DecisionTree::FindAttriNumByName(string attri) {
	for (int i = 0; i < MAXLEN; i++) {
		if (!state[0][i].compare(attri)) return i;
	}

	cerr << "can't find the numth of attribute" << endl;
	return 0;
}

//�ҳ�������ռ��������/����  
string DecisionTree::MostCommonLabel(vector <vector <string> > remain_state) {
	int p = 0, n = 0;
	for (unsigned i = 0; i < remain_state.size(); i++) {
		if (!remain_state[i][MAXLEN - 1].compare(yes)) p++;
		else n++;
	}
	if (p >= n) return yes;
	else return no;
}

//�ж������Ƿ������Զ�Ϊlabel  
bool DecisionTree::AllTheSameLabel(vector <vector <string> > remain_state, string label) {
	int count = 0;
	for (unsigned int i = 0; i < remain_state.size(); i++) {
		if (!remain_state[i][MAXLEN - 1].compare(label)) count++;
	}
	if (count == remain_state.size() - 1) return true;
	else return false;
}

//������Ϣ���棬DFS����������
Node * DecisionTree::BulidDecisionTreeDFS(Node * p, vector <vector <string> > remain_state,
	vector <string> remain_attribute) {
	if (p == NULL)
		p = new Node();

	if (AllTheSameLabel(remain_state, yes)) {
		p->attribute = yes;
		return p;
	}

	if (AllTheSameLabel(remain_state, no)) {
		p->attribute = no;
		return p;
	}

	double min_gini = 999, temp_gini;
	vector<string>::iterator min_it = remain_attribute.begin();
	vector<string>::iterator it1 = remain_attribute.begin();
	vector<double> gini(ComputeGini(remain_state, remain_attribute));
	int j = 0;
	for (int i = 1; i < remain_attribute.size() - 1; i++) {
		//temp_gini = gini[i - 1];
		if (gini[i - 1] < min_gini) {
			min_gini = gini[i - 1];
			
			while (j < i) {
				it1++;
				j++;
			}
			min_it = it1;
		}
	}
	

	// �������min_itָ������������ֵ�ǰ�������������������Լ�
	vector<string> new_attribute;
	vector<vector<string>> new_state;
	for (vector<string>::iterator it2 = remain_attribute.begin();
		it2 < remain_attribute.end(); it2++) {
		//string���ͱȽ�
		//��ǰ���ԵĻ���ϵ��������С����Ҫ�����ٱ�����
		if ((*it2).compare(*min_it)) new_attribute.push_back(*it2);
	}

	p->attribute = *min_it;
	vector<string> values = map_attribute_values[*min_it];
	int attribute_num = FindAttriNumByName(*min_it);
	new_state.push_back(attribute_row);
	for (vector<string>::iterator it3 = values.begin(); it3 < values.end(); it3++) {
		for (unsigned int i = 1; i < remain_state.size(); i++) {
			//�����ࡱ������ȡ��������ͬ����ֵ��ͬ��
			if (!remain_state[i][attribute_num].compare(*it3)) {
				new_state.push_back(remain_state[i]);
			}
		}
		Node * new_node = new Node();
		new_node->arrived_value = *it3;
		if (new_state.size() == 0) {
			new_node->arrived_value = MostCommonLabel(remain_state);
		}
		else
			BulidDecisionTreeDFS(new_node, new_state, new_attribute);

		p->childs.push_back(new_node);
		new_state.erase(new_state.begin() + 1, new_state.end());
	}

	return p;
}

void DecisionTree::Input() {
	string s;
	while (cin >> s, s.compare("end") != 0) {//-1Ϊ�������  
		item[0] = s;
		for (int i = 1; i < MAXLEN; i++) {
			cin >> item[i];
		}
		state.push_back(item);//ע��������ϢҲ�����ȥ��������  
	}
	for (int j = 0; j < MAXLEN; j++) {
		attribute_row.push_back(state[0][j]);
	}
}

void DecisionTree::PrintTree(Node *p, int depth) {
	//cout << "h" << p->attribute << endl;
	for (int i = 0; i < depth; i++) cout << '\t';//����������������tab  
	if (!p->arrived_value.empty()) {
		cout << p->arrived_value << endl;
		for (int i = 0; i < depth + 1; i++) cout << '\t';//����������������tab  
	}
	cout << p->attribute << endl;
	//cout << p->childs->attribute << endl;
	for (vector<Node*>::iterator it = p->childs.begin(); it != p->childs.end(); it++) {
		PrintTree(*it, depth + 1);
	}
}

void DecisionTree::FreeTree(Node *p) {
	if (p == NULL)
		return;
	for (vector<Node*>::iterator it = p->childs.begin(); it != p->childs.end(); it++) {
		FreeTree(*it);
	}
	delete p;
	tree_size++;
}

int main() {
	
	vector<vector<string>> state;
	DecisionTree* dt = new DecisionTree(state, 0);
	dt->Input();
	dt->ComputeMapFrom2DVector();//Ϊmap_attribute_values��ֵ

	long long freq, head, tail;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	
	//�ظ���ʱ��߾�ȷ��
	for (int i = 0; i < 20; i++) {
		Node* root = new Node();
		//start = clock();
		QueryPerformanceCounter((LARGE_INTEGER *)&head);

		dt->BulidDecisionTreeDFS(root, dt->GetState(), dt->GetAttribute());
		
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		cout << "the " << i + 1 << " time for a tree: " << (tail - head) * 1000 / freq << "ms" << endl;

		dt->FreeTree(root);
	}
	
	//dt->PrintTree(root, 1);
	return 0;
}