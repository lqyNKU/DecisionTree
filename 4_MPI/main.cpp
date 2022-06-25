#include "DecisionTree.h"
#include<iostream>
#include<ctime>
#include<time.h>
#include<stdlib.h>
#include<semaphore.h>
#include<mpi.h>
#include<malloc.h>

using namespace std;

vector<vector<string>> state;
DecisionTree* dt = new DecisionTree(state, 0);
long long freq,head,tail;
vector<vector <string>> state_t;
vector<string> attribute_t;
vector<vector<int>> count_sum;//每一个属性下有四个值
vector<int> proportion_sum;//每个属性对应一个
vector<double> entropy_sum;
int NUM_THREAD=4;//定义线程数
int NUM_COUNT=0;//定义需要计算的属性的个数
int seg = NUM_COUNT/NUM_THREAD;//定义每个线程的工作量


double timeforextra = 0;

class threadParam_t{
public:
    int id;//thread的id
    threadParam_t(int id){
        this->id = id;
    }
};

//信号量声明
sem_t sem_parent;
sem_t sem_children;

//互斥量声明
pthread_mutex_t * lock = new pthread_mutex_t();


void* threadFunc(void *param) {
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p->id;

    vector<double> entropy;
    for(int i = t_id*seg; (i < (t_id + 1)*seg) && (i < NUM_COUNT); i++){
        vector<string>::iterator it = attribute_t.begin();

        for (int j = 1; j < MAXLEN; j++) {
            //每次的remain_attribute包含开头的计数列和末尾的标签列，需去除
            for (it = attribute_t.begin() + 1; it < attribute_t.end() - 1; it++) {
                //找到对应属性
                if (!dt->attribute_row[j].compare(*it)) {
                    int proportion = 0;
                    vector<string>::iterator it_value = dt->map_attribute_values[*it].begin();
                    vector<int> tmp_count(4, 0);//初始化count，含有四个0，前二和后二分别对应一、二类
                    //cout << count[0] << endl;
                    for (int u = 1; u < state_t.size(); u++) {
                        //若此处属性值为value
                        if (!state_t[u][j].compare(*it_value)) {
                            proportion++;
                            //yes和no分别为两个label
                            if (!state_t[u][MAXLEN - 1].compare(dt->yes)) {
                                tmp_count[0] ++;
                            }
                            else tmp_count[1] ++;
                        }
                        //二分类，只有两种可能
                        else {
                            if (!state_t[u][MAXLEN - 1].compare(dt->yes)) {
                                tmp_count[2] ++;
                            }
                            else tmp_count[3] ++;
                        }
                    }
                    int sum1 = 0;
                    int sum3 = 0;

                    double p1, p3;
                    sum1 = tmp_count[0] + tmp_count[1];
                    sum3 = tmp_count[2] + tmp_count[3];

                    p1 = (double)tmp_count[0] / (double)sum1;
                    p3 = (double)tmp_count[2] / (double)sum3;

                    double tmp_entropy = (double)proportion / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
                    tmp_entropy += (1 - (double) proportion / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));
                    entropy.push_back(tmp_entropy);
                    break;
                }
            }
        }
    }

    pthread_mutex_lock(lock);//上锁
    //printf("Push back from thread%d.\n", t_id);
    for(int i = 0; i < entropy.size(); i++){
        entropy_sum.push_back(entropy[i]);
    }
    pthread_mutex_unlock(lock);//解锁

    sem_post(&sem_parent);
    return NULL;
}

//Pthreads版本
vector<double> DecisionTree::ComputeEntropyPthreads() {
    //清空
    vector<double> ().swap(entropy_sum);
	vector<vector<int>> ().swap(count_sum);
	vector<int> ().swap(proportion_sum);
    //信号量初始化
    sem_init(&sem_parent, 0, 0);
	sem_init(&sem_children, 0, 0);
    pthread_mutex_init(lock, NULL);

    //改变相应参数
    if(attribute_t.size() < 4){
        NUM_THREAD = attribute_t.size();
    }
    NUM_COUNT = attribute_t.size();
    seg = NUM_COUNT/NUM_THREAD;

    pthread_t* thread_handles = new pthread_t[NUM_THREAD];

    for(int i = 0; i < NUM_THREAD; i++){
        threadParam_t* newthread = new threadParam_t(i);
        clock_t t1 = clock();
        pthread_create(&thread_handles[i], NULL, threadFunc, (void*)newthread);
        clock_t t2 = clock();
        timeforextra+=(t2-t1)/float(CLOCKS_PER_SEC);
    }

    for (int i = 0; i<NUM_THREAD; i++)
	{
		sem_wait(&sem_parent);
	}

    for (int i=0; i<NUM_THREAD; i++)
	{
		pthread_join(thread_handles[i], NULL);
	}

	NUM_THREAD = 4;

    sem_destroy(&sem_parent);
	sem_destroy(&sem_children);
	return entropy_sum;
}
//Pthread: 根据具体属性和值来计算某属性值下基尼值
vector<double> DecisionTree::ComputeGiniPthread(vector<vector <string>> remain_state,
	vector<string> remain_attribute) {
    //清空上次的数据
    vector<vector<string>> ().swap(state_t);
    vector<string> ().swap(attribute_t);

    for(int i = 0;i < remain_state.size();i++){
        state_t.push_back(remain_state[i]);
    }

    for(int i = 0;i < remain_attribute.size();i++){
        attribute_t.push_back(remain_attribute[i]);
    }

	vector<double> entropy(ComputeEntropyPthreads());

	return entropy;

}

vector<vector<string>> DecisionTree::GetState() {
	return this->state;
}

vector<string> DecisionTree:: GetAttribute() {
	return this->attribute_row;
}

//构建树，初始化
DecisionTree::DecisionTree(vector<vector<string>> state, int tree_size) {
	this->state = state;
	this->tree_size = tree_size;
}

//根据数据实例计算属性与值组成的map
void DecisionTree::ComputeMapFrom2DVector() {
    unsigned int i, j, k;
	bool exited = false;
	vector<string> values;//存储该属性下的所有不重复的可能取值
	for (i = 1; i < MAXLEN - 1; i++) {//按照列遍历
		for (j = 1; j < state.size(); j++) {
			for (k = 0; k < values.size(); k++) {
				//string的比较，相等为0，不相等为-1
				//防止重复值出现
				if (!values[k].compare(state[j][i])) exited = true;
			}
			if (!exited) {
				values.push_back(state[j][i]);//注意Vector的插入都是从前面插入的，注意更新it，始终指向vector头
			}
			exited = false;
		}
		map_attribute_values[state[0][i]] = values;//将属性state[0][i]与values对应形成一个map
		values.erase(values.begin(), values.end());
	}
}

//计数用于传给ComputeGini
vector<double> DecisionTree::ComputeEntropy() {
	for (int i = 0; i < count_sum.size(); i++) {
		int sum1 = 0;
		int sum3 = 0;
		int pro = proportion_sum[i];
		double p1, p3;
		sum1 = count_sum[i][0] + count_sum[i][1];
		sum3 = count_sum[i][2] + count_sum[i][3];
		p1 = (double)count_sum[i][0] / (double)sum1;
		p3 = (double)count_sum[i][2] / (double)sum3;
		double entropy = (double)pro / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
		entropy += (1 - (double) pro / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));
		entropy_sum.push_back(entropy);
	}
	return entropy_sum;


}

vector<double> DecisionTree::f(vector<vector <string>> remain_state,
	vector<string> remain_attribute){
    vector<double> entropy;
    for(int i = 1; i < remain_attribute.size() - 1; i++){
        vector<string>::iterator it = remain_attribute.begin();
        for (int j = 1; j < MAXLEN; j++) {
            //每次的remain_attribute包含开头的计数列和末尾的标签列，需去除
            for (it = remain_attribute.begin() + 1; it < remain_attribute.end() - 1; it++) {
                //找到对应属性
                if (!attribute_row[j].compare(*it)) {
                    int proportion = 0;
                    vector<string>::iterator it_value = map_attribute_values[*it].begin();
                    vector<int> tmp_count(4, 0);//初始化count，含有四个0，前二和后二分别对应一、二类
                    //cout << count[0] << endl;
                    for (int u = 1; u < remain_state.size(); u++) {
                        //若此处属性值为value
                        if (!remain_state[u][j].compare(*it_value)) {
                            proportion++;
                            //yes和no分别为两个label
                            if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                tmp_count[0] ++;
                            }
                            else tmp_count[1] ++;
                        }
                        //二分类，只有两种可能
                        else {
                            if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                tmp_count[2] ++;
                            }
                            else tmp_count[3] ++;
                        }
                    }
                    int sum1 = 0;
                    int sum3 = 0;

                    double p1, p3;
                    sum1 = tmp_count[0] + tmp_count[1];
                    sum3 = tmp_count[2] + tmp_count[3];

                    p1 = (double)tmp_count[0] / (double)sum1;
                    p3 = (double)tmp_count[2] / (double)sum3;

                    double tmp_entropy = (double)proportion / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
                    tmp_entropy += (1 - (double) proportion / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));
                    entropy.push_back(tmp_entropy);
                    break;
                }
            }
        }
    }
    return entropy;
}

vector<double> DecisionTree::ComputeGiniMPI(vector<vector <string>> remain_state, vector <string> remain_attribute){
    int N = remain_attribute.size() - 2;//去掉首尾属性
    vector<double> entropy;
    entropy.resize(N);
    double tmp[N];

    clock_t t1 = clock();
    MPI_Init(NULL, NULL);
    clock_t t2 = clock();
    timeforextra += (t2 - t1)/float(CLOCKS_PER_SEC);

    MPI_Status status[N];
    int my_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &my_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int my_tag = 99;

    int seg = N/my_size;//每个节点工作量

    if(seg < 1){
        seg = 1;
    }

    int start, finish;
    start = 1 + seg*my_rank;
    finish = 1 + seg*(my_rank + 1);

    if(my_rank!=0){
            double entropy_thread[finish - start];
            for(int i = start; i < finish; i++){
            vector<string>::iterator it = remain_attribute.begin();
            for (int j = 1; j < MAXLEN; j++) {
                //每次的remain_attribute包含开头的计数列和末尾的标签列，需去除
                for (it = remain_attribute.begin() + 1 + seg * my_rank; it < remain_attribute.begin() + 1 + seg * (my_rank + 1); it++) {
                    //找到对应属性
                    if (!attribute_row[j].compare(*it)) {
                        int proportion = 0;
                        vector<string>::iterator it_value = map_attribute_values[*it].begin();
                        vector<int> tmp_count(4, 0);//初始化count，含有四个0，前二和后二分别对应一、二类
                        //cout << count[0] << endl;
                        for (int u = 1; u < remain_state.size(); u++) {
                            //若此处属性值为value
                            if (!remain_state[u][j].compare(*it_value)) {
                                proportion++;
                                //yes和no分别为两个label
                                if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                    tmp_count[0] ++;
                                }
                                else tmp_count[1] ++;
                            }
                            //二分类，只有两种可能
                            else {
                                if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                    tmp_count[2] ++;
                                }
                                else tmp_count[3] ++;
                            }
                        }
                        int sum1 = 0;
                        int sum3 = 0;

                        double p1, p3;
                        sum1 = tmp_count[0] + tmp_count[1];
                        sum3 = tmp_count[2] + tmp_count[3];

                        p1 = (double)tmp_count[0] / (double)sum1;
                        p3 = (double)tmp_count[2] / (double)sum3;

                        double tmp_entropy = (double)proportion / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
                        tmp_entropy += (1 - (double) proportion / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));

                        cout<<"entropy from "<<my_rank<<": "<<tmp_entropy<<endl;
                        entropy_thread[i-start] = tmp_entropy;

                        break;
                    }
                }
            }
        }
        //发送给0号进程
        MPI_Send(&entropy_thread[0], seg, MPI_DOUBLE, 0, my_tag, MPI_COMM_WORLD);
    }
    else/*my_rank=0*/{
        for(int i = start; i < finish; i++){
            vector<string>::iterator it = remain_attribute.begin();
            for (int j = 1; j < MAXLEN; j++) {
                //每次的remain_attribute包含开头的计数列和末尾的标签列，需去除
                for (it = remain_attribute.begin() + 1 + seg * my_rank; it < remain_attribute.begin() + 1 + seg * (my_rank + 1); it++) {
                    //找到对应属性
                    if (!attribute_row[j].compare(*it)) {
                        int proportion = 0;
                        vector<string>::iterator it_value = map_attribute_values[*it].begin();
                        vector<int> tmp_count(4, 0);//初始化count，含有四个0，前二和后二分别对应一、二类
                        //cout << count[0] << endl;
                        for (int u = 1; u < remain_state.size(); u++) {
                            //若此处属性值为value
                            if (!remain_state[u][j].compare(*it_value)) {
                                proportion++;
                                //yes和no分别为两个label
                                if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                    tmp_count[0] ++;
                                }
                                else tmp_count[1] ++;
                            }
                            //二分类，只有两种可能
                            else {
                                if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                    tmp_count[2] ++;
                                }
                                else tmp_count[3] ++;
                            }
                        }
                        int sum1 = 0;
                        int sum3 = 0;

                        double p1, p3;
                        sum1 = tmp_count[0] + tmp_count[1];
                        sum3 = tmp_count[2] + tmp_count[3];

                        p1 = (double)tmp_count[0] / (double)sum1;
                        p3 = (double)tmp_count[2] / (double)sum3;

                        double tmp_entropy = (double)proportion / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
                        tmp_entropy += (1 - (double) proportion / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));

                        cout<<"entropy from "<<my_rank<<": "<<tmp_entropy<<endl;

                        tmp[i-1]=tmp_entropy;

                        break;
                    }
                }
            }
        }
        //多余属性未处理
        if(seg*(my_size)>N){
            for(int i = seg*my_size; i < N+1; i++){
            vector<string>::iterator it = remain_attribute.begin();
            for (int j = 1; j < MAXLEN; j++) {
                //每次的remain_attribute包含开头的计数列和末尾的标签列，需去除
                for (it = remain_attribute.begin() + 1 + seg * my_size; it < remain_attribute.end() - 1; it++) {
                    //找到对应属性
                    if (!attribute_row[j].compare(*it)) {
                        int proportion = 0;
                        vector<string>::iterator it_value = map_attribute_values[*it].begin();
                        vector<int> tmp_count(4, 0);//初始化count，含有四个0，前二和后二分别对应一、二类
                        //cout << count[0] << endl;
                        for (int u = 1; u < remain_state.size(); u++) {
                            //若此处属性值为value
                            if (!remain_state[u][j].compare(*it_value)) {
                                proportion++;
                                //yes和no分别为两个label
                                if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                    tmp_count[0] ++;
                                }
                                else tmp_count[1] ++;
                            }
                            //二分类，只有两种可能
                            else {
                                if (!remain_state[u][MAXLEN - 1].compare(yes)) {
                                    tmp_count[2] ++;
                                }
                                else tmp_count[3] ++;
                            }
                        }
                        int sum1 = 0;
                        int sum3 = 0;

                        double p1, p3;
                        sum1 = tmp_count[0] + tmp_count[1];
                        sum3 = tmp_count[2] + tmp_count[3];

                        p1 = (double)tmp_count[0] / (double)sum1;
                        p3 = (double)tmp_count[2] / (double)sum3;

                        double tmp_entropy = (double)proportion / (sum1 + sum3)*(1 - p1 * p1 - (1 - p1)*(1 - p1));
                        tmp_entropy += (1 - (double) proportion / (sum1 + sum3))*(1 - p3 * p3 - (1 - p3)*(1 - p3));

                        tmp[i-1]=tmp_entropy;

                        break;
                    }
                }
            }
        }
        }
        //用mpi_any_source防止阻塞
        for(int i = 1; i < my_size; ++i){
            MPI_Recv(&tmp[seg*i], seg, MPI_DOUBLE, MPI_ANY_SOURCE, my_tag, MPI_COMM_WORLD, &status[i]);
        }
        for(int i = 0; i < N; ++i){
        entropy.push_back(tmp[i]);
        }
    }
    t1 = clock();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    t2 = clock();
    timeforextra += (t2 - t1)/float(CLOCKS_PER_SEC);
    return entropy;

}

//根据具体属性和值来计算某属性值下基尼值
vector<double> DecisionTree::ComputeGini(vector<vector <string>> remain_state,
	vector<string> remain_attribute) {

    vector<double> ().swap(entropy_sum);

	vector<vector<int>> ().swap(count_sum);
	vector<int> ().swap(proportion_sum);

	unsigned int i, j;
	bool done_flag = false;


	vector<string>::iterator it = remain_attribute.begin();

	for (j = 1; j < MAXLEN; j++) {
		//每次的remain_attribute包含开头的计数列和末尾的标签列，需去除
		for (it = remain_attribute.begin() + 1; it < remain_attribute.end() - 1; it++) {
			//找到对应属性
			if (!attribute_row[j].compare(*it)) {
				int proportion = 0;
				vector<string>::iterator it_value = map_attribute_values[*it].begin();
				vector<int> tmp_count(4, 0);//初始化count，含有四个0，前二和后二分别对应一、二类
				//cout << count[0] << endl;
				for (i = 1; i < remain_state.size(); i++) {
					//若此处属性值为value
					if (!remain_state[i][j].compare(*it_value)) {
						proportion++;
						//yes和no分别为两个label
						if (!remain_state[i][MAXLEN - 1].compare(yes)) {
							tmp_count[0] ++;
						}
						else tmp_count[1] ++;
					}
					//二分类，只有两种可能
					else {
						if (!remain_state[i][MAXLEN - 1].compare(yes)) {
							tmp_count[2] ++;
						}
						else tmp_count[3] ++;
					}
				}
				count_sum.push_back(tmp_count);
				proportion_sum.push_back(proportion);
			}
		}
	}

	vector<double> entropy(ComputeEntropy());
	//vector<double> entropy(ComputeEntropyPthreads());

	return entropy;

}

int DecisionTree::FindAttriNumByName(string attri) {
	for (int i = 0; i < MAXLEN; i++) {
		if (!state[0][i].compare(attri)) return i;
	}

	cerr << "can't find the numth of attribute" << endl;
	return 0;
}

//找出样例中占多数的正/负性
string DecisionTree::MostCommonLabel(vector <vector <string> > remain_state) {
	int p = 0, n = 0;
	for (unsigned i = 0; i < remain_state.size(); i++) {
		if (!remain_state[i][MAXLEN - 1].compare(yes)) p++;
		else n++;
	}
	if (p >= n) return yes;
	else return no;
}

//判断样例是否正负性都为label
bool DecisionTree::AllTheSameLabel(vector <vector <string> > remain_state, string label) {
	int count = 0;
	for (unsigned int i = 0; i < remain_state.size(); i++) {
		if (!remain_state[i][MAXLEN - 1].compare(label)) count++;
	}
	if (count == remain_state.size() - 1) return true;
	else return false;
}

//计算信息增益，DFS构建决策树
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
    //串行版本
    vector<double> gini(f(remain_state, remain_attribute));
    //并行版本
    //ComputeGiniMPI(remain_state, remain_attribute);
    //vector<double> gini(ComputeGiniMPI(remain_state, remain_attribute));

	int j = 0;
	for (int i = 1; i < remain_attribute.size() - 1; i++) {
		if (gini[i - 1] < min_gini) {
			min_gini = gini[i - 1];
			while (j < i) {
				it1++;
				j++;
			}
			min_it = it1;
		}
	}

	// 下面根据min_it指向的属性来划分当前样例，更新例集和属性集
	vector<string> new_attribute;
	vector<vector<string>> new_state;
	for (vector<string>::iterator it2 = remain_attribute.begin();
		it2 < remain_attribute.end(); it2++) {
		//string类型比较
		//当前属性的基尼系数不是最小，需要后续再被划分
		if ((*it2).compare(*min_it)) new_attribute.push_back(*it2);
	}

	p->attribute = *min_it;
	vector<string> values = map_attribute_values[*min_it];
	int attribute_num = FindAttriNumByName(*min_it);
	new_state.push_back(attribute_row);
	for (vector<string>::iterator it3 = values.begin(); it3 < values.end(); it3++) {
		for (unsigned int i = 1; i < remain_state.size(); i++) {
			//“分类”，依次取出具有相同属性值的同类
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
    //随机生成2000组数据
	//首行
	for(int i = 0; i < MAXLEN - 1; i++){
        item[i] = 'A' + i;
	}
	item[MAXLEN - 1]= "label";
	state.push_back(item);
	for(int i = 1; i <= 100; i++){
        item[0]=std::to_string(i);
        for(int j = 1; j < MAXLEN - 1; j++){

        }
	}

	for(int i = 1; i <= 2000; i++){
        for(int j = 1; j < MAXLEN - 1; j++){
            item[j] = std::to_string(rand()%2);
        }
        if(rand()%2==0){
            item[MAXLEN - 1] = "yes";
        }else{
            item[MAXLEN - 1] = "no";
        }
        state.push_back(item);
	}
	for (int j = 0; j < MAXLEN; j++) {
		attribute_row.push_back(state[0][j]);
	}
}

void DecisionTree::PrintTree(Node *p, int depth) {
	for (int i = 0; i < depth; i++) cout << '\t';//按照树的深度先输出tab
	if (!p->arrived_value.empty()) {
		cout << p->arrived_value << endl;
		for (int i = 0; i < depth + 1; i++) cout << '\t';//按照树的深度先输出tab
	}
	cout << p->attribute << endl;
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
	dt->Input();
	dt->ComputeMapFrom2DVector();//为map_attribute_values赋值

	//重复计时提高精确性
	for (int i = 0; i < 20; i++) {
		Node* root = new Node();
        NUM_THREAD=4;
        timeforextra=0;
        clock_t t1 = clock();
		dt->BulidDecisionTreeDFS(root, dt->GetState(), dt->GetAttribute());
        clock_t t2 = clock();

        cout<<"time for a tree(including extra time): "<<(t2-t1)/float(CLOCKS_PER_SEC)<<endl;
        cout<<"time for pure building:"<<timeforextra<<"s"<<endl;
        cout<<"finish building the tree"<< i + 1<<endl<<endl;

		dt->FreeTree(root);
	}

	return 0;
}

