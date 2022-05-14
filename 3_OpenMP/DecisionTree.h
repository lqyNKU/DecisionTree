#ifndef DECISIONTREE_H_INCLUDED
#define DECISIONTREE_H_INCLUDED
#ifndef ML_DECISION_H//如果这个宏没有被定义
#define ML_DECISION_H//则定义宏

#include "DecisionTreeStruct.h"
#include <string>
#define MAXLEN 12 //输入每行的数据个数

class DecisionTree {
public:
	__declspec(dllexport) DecisionTree(vector<vector<string>> state, int tree_size);
	__declspec(dllexport) void ComputeMapFrom2DVector();
	__declspec(dllexport) vector<double> ComputeEntropy();
	__declspec(dllexport) vector<double> ComputeEntropyParallel(vector<vector<int>> numOfEachAttr, vector<int>proportionOfEachAttr);
	__declspec(dllexport) vector<double> ComputeEntropyPthreads();
	__declspec(dllexport) vector<double> ComputeGini(vector <vector<string>> remain_state, vector<string> remain_attribute);
    __declspec(dllexport) vector<double> ComputeGiniPthread(vector<vector <string>> remain_state, vector<string> remain_attribute);
    __declspec(dllexport) friend void *threadFunc(void *param);
    __declspec(dllexport) vector<double> f(vector<vector <string>> remain_state,vector<string> remain_attribute);
	//__declspec(dllexport) double ComputeGain(vector <vector <string> > remain_state, string attribute);

	//__declspec(dllexport) double Compute(int n, int sum);
	__declspec(dllexport) vector<string> GetAttribute();
	__declspec(dllexport) vector<vector<string>> GetState();

	__declspec(dllexport) int FindAttriNumByName(string attri);
	__declspec(dllexport) string MostCommonLabel(vector <vector <string> > remain_state);
	__declspec(dllexport) bool AllTheSameLabel(vector <vector <string> > remain_state, string label);
	__declspec(dllexport) Node * BulidDecisionTreeDFS(Node * p, vector <vector <string> > remain_state, vector <string> remain_attribute);
	__declspec(dllexport) void Input();
	__declspec(dllexport) void PrintTree(Node *p, int depth);
	__declspec(dllexport) void FreeTree(Node *p);

private:
	vector <vector <string> > state;//实例集
	vector <string> item{ MAXLEN };//对应一行实例集
	vector <string> attribute_row;//保存首行即属性行数据
	string end = "end";//输入结束
	string yes = "yes";
	string no = "no";
	string blank = "";
	map<string, vector < string > > map_attribute_values;//存储属性对应的所有的值
	int tree_size = 0;//几个结点
};
#endif



#endif // DECISIONTREE_H_INCLUDED
