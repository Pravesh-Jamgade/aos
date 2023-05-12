#include<bits/stdc++.h>
#include<functional>
using namespace std;
#define vi vector<int>
#define fi first 
#define se second
#define pb push_back
#define pii pair<int,int>
#define ar array
#define Edge pair<int,int>
#define EDGE(u,v) make_pair(u,v)
typedef long long ll;
#define DEBUG

class Com{
	public:
	int processor;
	double cost;
};

class Node{
	public:
	vector<Com> data;
};

class Graph{
public:
	int TT[MAX][MAX] = {0};
	vector<Edge> pgraph[MAX];
	vector<Edge> sgraph[MAX];
	vector<Edge> pgraph2[MAX], sgraph2[MAX];
	double t2p_cost[MAX][MAX], t2p_cost2[MAX][MAX];
	vector<double> est[MAX], ect[MAX], fproc[MAX];
	vector<bool> vis;
	vector<int> cpred;
	vector<int> cpt[MAX];
	vector<double> level;
	vector<int> task_2_processor;
	vector<int> processor_2_task;
	vector<int> t2p[MAX];
	vector<int> p2t[MAX];
	// level array sorted in increasing order
	int entry_task = -1; // last (highest level value node (entry node of DAG))
	int exit_task = -1;// first (lowest level value node (exit node of DAG))
	int extra_node = -1;

	int nodes, edges, processor;
	queue<int> bfs_queue;
	Graph(int nodes, int edges, int processor){
		this->nodes = nodes, this->edges = edges, this->processor = processor;
		vis.resize(nodes+1,false);
		cpred.resize(nodes+1,-1);
		level.resize(nodes+1,0);
		task_2_processor.resize(MAX,-1);
		processor_2_task.resize(processor+1,-1);
		extra_node = nodes+10;
	}
	void take_input();

	void test();
	void test2();
	void clean_copy_graph();
	void copy_graph();

	void doBFS(int node, int parent);
	void doBFS1(int node);
	bool doBFS2(int node, vector<int>& trail);
	bool doDFS(int node, double& sum, vector<int>& path);
	void get_topo_order_of_task_on_processor(int parentProc, int task, int parent_task, vector<int>& topo, list<int>* sch);
	int makespan(list<int>* sch);
	void gen_schedules(list<int>* schedules);
	void update_task2processor(list<int>sch[MAX]);
	void copy_sch(list<int>* copied, list<int>* sch);
	void verify_sch(list<int>* schedules);
	void copy_between_start_to_element(list<int>& new_schedule, vector<int> schedule, int task);
	bool on_same_cluster(int parent, int child, list<int>* sch);
	vector<int> find_curr_allocated_processor(int task, list<int>* sch);
	void update_sch(list<int>* sch, list<int>* newSch);

	void compute_parameter();
	void task_clustering();
	void task_duplication(list<int>* sch);
	void processor_merging(list<int>* sch);
};

void Graph::take_input(){
	t2p_cost[MAX][MAX] = {0};

	// edge - comm cost
	for(int i=0; i< edges; i++){
		int u,v;
		double cost;
		cin >> u >> v >> cost;			  // input u->v edge
		sgraph[u].push_back(Edge(cost,v));// u={v} successor
		pgraph[v].push_back(Edge(cost,u));// v={u} predecessor
		TT[u][v] = TT[v][u] = cost;
	}

	// compute cost of task node i on processor k
	for(int i=1; i<= nodes; i++){
		int node; cin>>node;
		for(int j=1; j<= processor; j++){
			int task_processor_cost;
			cin>> task_processor_cost;
			t2p_cost[i][j]=task_processor_cost;
		}
	}

	for(int i=1; i<= nodes; i++){
		est[i].resize(processor+1, 0);
		ect[i].resize(processor+1, 0);
		fproc[i].resize(processor+1, 0);
	}
}

void Graph::test(){
	cout << "***********************************\n";
	for(int i=1; i<= nodes; i++){
		cout << i << ":";
		for(auto node: sgraph[i]){
			cout << node.second << ",";
		}
		cout << '\n';
	}
	cout << "***********************************\n";
	for(int i=1; i<= nodes; i++){
		for(int j=1; j<= processor; j++){
			cout << t2p_cost[i][j] << ",";
		}
		cout << '\n';
	}
}

void Graph::test2(){
	cout << "***********************************\n";
	for(int i=1; i<= extra_node; i++){
		cout << i << ":";
		for(auto node: sgraph2[i]){
			cout <<  "(" << node.fi << "," << node.se << "),";
		}
		cout << '\n';
	}
	cout << "***********************************\n";
	for(int i=1; i<= extra_node; i++){
		cout << i << ": "; 
		for(int j=1; j<= processor; j++){
			cout << t2p_cost[i][j] << ",";
		}
		cout << '\n';
	}
}

/*calculate est, ect, fproc for each task of DAG*/
void Graph::doBFS(int node, int parent){

	bfs_queue.push(node);

	while(!bfs_queue.empty()){
		int u = bfs_queue.front();
		bfs_queue.pop();
		if(vis[u]) continue;
		vis[u] = true;

		if(pgraph[u].size() == 0){
			// for root node
			for(int i=1; i<= processor; i++){
				est[u][i] =0;
				ect[u][i] = est[u][i] + t2p_cost[u][i];
			}

		}
		else{
			for(int i=1; i<= processor; i++){
				// for nodes having predecessor
				for(auto edge: pgraph[u]){ //v->u (v is parent of u)
					int predecessor = edge.se;
					double cost_of_edge = edge.fi;
					int first_fproc_of_predecessor = fproc[predecessor][1];// first fav processor 
					double ect_of_predecessor_on_first_fproc = ect[predecessor][first_fproc_of_predecessor];
					double a = ect_of_predecessor_on_first_fproc + cost_of_edge;					
					double ect_of_predecessor_on_processor_i = ect[predecessor][i];	
					double min_est = min(a, ect_of_predecessor_on_processor_i);
					est[u][i] = max(min_est, est[u][i]);
				}
				ect[u][i] = est[u][i] + t2p_cost[u][i];
			}
		}
		
		// find fproc for task u
		vector<pair<double, int>> ect_processor_pair(processor);
		for(int i=1; i<= processor; i++){
			ect_processor_pair[i-1]=EDGE(ect[u][i], i);
		}

		sort(ect_processor_pair.begin(), ect_processor_pair.end());

		// set fav processor for task 
		for(int i=1; i<= processor; i++){
			fproc[u][i] = ect_processor_pair[i-1].se;
		}

		for(auto edge: sgraph[u]){
			int v = edge.se;
			bfs_queue.push(v);
		}
	}
}

/*calculate critical predecessor for each task of DAG*/
void Graph::doBFS1(int node){
	while(!bfs_queue.empty()){
		bfs_queue.pop();
	}
	
	fill(vis.begin(), vis.end(), false);
	bfs_queue.push(node);
	while(!bfs_queue.empty()){
		int u = bfs_queue.front();
		bfs_queue.pop();
		if(vis[u]) continue;
		vis[u] = true;

		if(pgraph[u].size() == 0){
			cpred[u]=0;
		}
		else{
			vector<pair<double, int>> cost2pred;
			for(auto edge: pgraph[u]){
				int predecessor = edge.se;
				double cost_of_edge = edge.fi;

				int first_fproc_of_predecessor = fproc[predecessor][1];// first fav processor 
				int first_fproc_of_current = fproc[u][1];
				
				double ect_of_predecessor_on_first_fproc = ect[predecessor][first_fproc_of_predecessor];
				double cost = 0;
				
				if(first_fproc_of_predecessor != first_fproc_of_current){
					cost = edge.fi;
				}
				
				cost = cost + ect_of_predecessor_on_first_fproc;
				cost2pred.pb({cost, predecessor});
			}
			sort(cost2pred.begin(), cost2pred.end());
			cpred[u] = cost2pred[cost2pred.size()-1].se;
		}
		for(auto edge: sgraph[u]){
			int v = edge.se;
			bfs_queue.push(v);
		}
	}
}

bool Graph::doBFS2(int node, vector<int>& trail){
	if(1 == node){
		trail.pb(1);
		return true;
	}
	
	if(doBFS2(cpred[node], trail)){
		trail.pb(node);
		return true;
	}
	return false;	
}

bool Graph::doDFS(int u, double& sum, vector<int>& path){
	if(u == nodes){
		return true;
	}
	for(auto edge: sgraph[u]){
		int v = edge.se;
		double c = edge.fi;
		if(vis[v]) continue;
		bool test = doDFS(v, sum, path);
		if(test){
			path.pb(v);
			sum += c;
			return true;
		}
	}
	return false;
}

// processor, parent, child, schedule
bool Graph::on_same_cluster(int parentProc, int child, list<int>* sch){
	if(parentProc == -1) return true;//this is root node of schedule

	// on parents processor do you see the child, if yes then belongs to same cluster
	for(auto e: sch[parentProc]){
		if(e==child) return true;
	}
	return false;
}

// processro of task, task, parent of task, collect_all_task in topo, current schedule
void Graph::get_topo_order_of_task_on_processor(int parentProc, int task, int parent, vector<int>& all_task, list<int>* sch){
	if(vis[task]) return;
	vis[task] = true;
	for(auto edge: sgraph[task]){
		if(on_same_cluster(parentProc, edge.se, sch)){
			get_topo_order_of_task_on_processor(parentProc, edge.se, task, all_task, sch);
			all_task.pb(edge.se);
		}
	}
}

void Graph::verify_sch(list<int>* schedules){
	cout << "verify schedules:\n";
	for(int i=1; i<= processor; i++){
		cout << i << ":";
		for(auto task: schedules[i]){
			cout << task << ",";
		}
		cout<<'\n';
	}
}

void Graph::gen_schedules(list<int>* schedules){
	for(int task=1; task<= nodes; task++){
		int proc = task_2_processor[task];
		if(proc == -1) continue;
		schedules[proc].pb(task);
	}

	for(int task=1; task<= nodes; task++){
		int proc = task_2_processor[task];
		if(schedules[proc].size() != 0){
			auto found = find(schedules[proc].begin(), schedules[proc].end(), entry_task);
			if(found==schedules[proc].end()){
				schedules[proc].push_front(entry_task);
			}
		}
	}
}

/*args: copy, original*/
void Graph::copy_sch(list<int>*copied, list<int>*sch){
	for(int i=1; i<= processor; i++){
		for(auto task: sch[i]){
			copied[i].push_back(task);
		}
	}
}

void Graph::clean_copy_graph(){
	for(int i=1; i<= nodes; i++){
		vector<pii>().swap(pgraph2[i]);
		vector<pii>().swap(sgraph2[i]);
	}
	for(int i=1; i<= nodes; i++){
		for(int j=1; j<= processor; j++){
			t2p_cost2[i][j]=0;
		}
	}
}

void Graph::copy_graph(){
	for(int i=1; i<= nodes; i++){
		for(auto e: pgraph[i]){
			pgraph2[i].push_back(e);
		}
	}

	for(int i=1; i<= nodes; i++){
		for(auto e: sgraph[i]){
			sgraph2[i].push_back(e);
		}
	}

	for(int i=1; i<= nodes; i++){
		for(int j=1; j<= processor; j++){
			t2p_cost2[i][j]=t2p_cost[i][j];
		}
	}
}

// int Graph::makespan2(list<int>* schedule){
// 	// build a graph from new schedule
// 	vector<Node> g[MAX];
// 	for(int i=1; i<= processor; i++){
// 		for(auto node: schedule[i]){

// 		}
// 	}
// }

int Graph::makespan(list<int>* all_schedules){
	assert(exit_task!=-1);

	vector<bool> done(nodes+1, false);
	// collect nodes shared between processors
	vector<int> shared(nodes+1, 1);
	
	vector<double> aaa(100, 0);

	// find shared nodes: <0 shared, ==0 not shared
	for(int i=1; i<= processor; i++){
		for(auto entry: all_schedules[i]){
			shared[entry]--;
		}
	}

	fill(task_2_processor.begin(), task_2_processor.end(), 0);
	update_task2processor(all_schedules);

	vector<vector<pii>> all_poss_cost_of_shared_nodes(nodes+1, vector<pii>());

	// search all processor and see if nodes is shared on that processor, add its cost
	for(int k=1; k<=nodes; k++){
		// shared node
		int sh_node = shared[k];
		if(sh_node>=0) continue;

		// check on all clusters
		for(int i=1; i<= processor; i++){
			for(auto e: all_schedules[i]){

				// shared node 'sh_node' , on processor i
				if(e == k){
					double cost_on_processor = t2p_cost2[k][i];
					// node: [node, processor]
					all_poss_cost_of_shared_nodes[k].push_back({k, i});
					// sh_task's will be replaced by new and extra task, hence this task will become obsolete
					task_2_processor[k] = 0;
					break;
				}
			}
		}
	}
	
	// for shared node: decompose them to number of processors sharing them
	// update the graph accordingly
	// copy t2p_cost as well
	copy_graph();

	for(int kp=1; kp<= nodes; kp++){
		auto shared_node = all_poss_cost_of_shared_nodes[kp];
		vector<int> newNodes;
		bool pg, sg;
		pg=sg=false;

		int realNode = kp;

		// erasing edge entries in existing graph for node, if it is shared across processors for computing 
		if(shared_node.size()){
			sgraph2[realNode] = vector<pii>();
			pgraph2[realNode] = vector<pii>();
		}

		//add extra node
		for(auto costAndProc: shared_node){
			int proc = costAndProc.se;
			int newNode = extra_node++;

			// original processor is assigned to newly created task
			task_2_processor[newNode] = proc;
			t2p_cost2[newNode][proc] = t2p_cost2[realNode][proc];

			//** just doing this for sgraph
			vector<Edge> immed_to_s = sgraph[realNode];
			vector<Edge> immed_to_p = pgraph[realNode];

			// immediate child's of realNode
			for(auto e: immed_to_s){
				// if child and realNode are on same processor
				if(on_same_cluster(proc, e.se, all_schedules)){
					sgraph2[newNode].push_back(e);
					sg=true;
					break;
				}
			}

			// immediate parent's of realNode
			for(auto e: immed_to_p){
				// if parent and realNode are on same processor
				if(on_same_cluster(proc, e.se, all_schedules)){
					sgraph2[e.se].push_back(EDGE(e.fi,newNode));
					pg=true;
					break;
				}
			}

			newNodes.push_back(newNode);

			// merge to zero cost and realNode
			if(sg and !pg)
				sgraph2[realNode].push_back(EDGE(0,newNode));
			if(!sg and pg)
				pgraph2[newNode].push_back(EDGE(0,realNode));		

			sg=pg=false;	
		}
	}
	// test2();
	// exit(0);

	// cout << "---------------------TASK 2 PROCESSOR-------------------------\n";
	// for(int i=1; i<= extra_node; i++){
	// 	cout << i <<":" << task_2_processor[i]<<'\n';
	// }
	// cout << "--------------------------------------------------------------\n";
	// for(int i=1; i<= extra_node; i++){
	// 	cout << i << ": "; 
	// 	for(int j=1; j<= processor; j++){
	// 		cout << t2p_cost[i][j] << ",";
	// 	}
	// 	cout << '\n';
	// }
	// cout << "--------------------------------------------------------------\n";

	queue<pii> q;
	q.push({entry_task, task_2_processor[entry_task]});
	
	int wt = 0;

	while(!q.empty()){
		int node = q.front().fi;
		int proc = q.front().se;
		q.pop();

		if(done[node]) continue;
		done[node]=true;

		double node_cost = t2p_cost2[node][proc];
		// cout << "[Parent] "<< node <<", node_cost:"<<node_cost<<", node:"<<node<<'\n'; 
		
		// all predecessors
		for(auto entry: sgraph2[node]){
			
			int child = entry.se;
			int child_proc = task_2_processor[child];
			double link_cost = entry.fi;
			if(child_proc == proc)
				link_cost=0;
			double cost = link_cost + node_cost + aaa[node];
			
			if(aaa[child] < cost){
				// cout << "[Child] "<<child<<", Parent-2-Child link_cost:" << link_cost <<", yes"<<'\n'; 
				aaa[child]=cost;
			}
			else{
				// cout << "[Child] "<< child <<"Parent-2-Child link_cost:" << link_cost << ", no" <<'\n';
			}
			q.push({child, child_proc});
		}
	}

	aaa[exit_task] += t2p_cost2[exit_task][task_2_processor[exit_task]];
	double ret = aaa[exit_task];
	if(aaa.size()>0) aaa.clear();
	for(auto e: all_poss_cost_of_shared_nodes){
		if(e.size()>0) e.clear();
	}

	// cout << '\n';
	// for(int i=1; i<=extra_node; i++){
	// 	cout<< i << ":" << aaa[i]<<'\n' ; //t2p_cost[i][task_2_processor[i]] << '\n';
	// }
	clean_copy_graph();
	extra_node = nodes+10;
	// cout << "-----------------------TASK TO PATH COST---------------------------------------\n";
	// for(int i=1; i<=extra_node; i++){
	// 	cout<< i << ":" << aaa[i]<<'\n' ; //t2p_cost[i][task_2_processor[i]] << '\n';
	// }
	// cout << "--------------------------------------------------------------\n";
	// for(auto e: aaa){cout<< e<< ",";}
	// exit(0);
	// cout << "makespan:"<<ret<<'\n';
	// verify_sch(all_schedules);
	return ret;
}

/*find which processor is allocated for curTask*/
vector<int> Graph::find_curr_allocated_processor(int currTask, list<int>* sch){
	vector<int> all_proc;
	for(int proc=1; proc<=processor; proc++){
		for(auto task: sch[proc]){
			if(task == currTask){
				all_proc.push_back(proc);
			}
		}
	}
	return all_proc;
}

void Graph::compute_parameter(){
	// assuming start vertex is node 1;
	doBFS(1, -1);
	doBFS1(1);

	// from each task to exit node path, find bottleneck processors cost
	// find path to exit node from curr node
	for(int i=1; i<= nodes; i++){
		fill(vis.begin(), vis.end(), false);
		// collect task which are part of path
		vector<int> path;

		double sum = 0;
		doDFS(i, sum, path);
		path.pb(i);
		// add path cost (comm cost)
		level[i]+=sum;
		// for each task on path, add their max cost of compute among their processors 
		for(auto v: path){
			double mnX = 0;
			for(int j=1; j<= processor; j++){
				mnX = max(t2p_cost[v][j], mnX);
			}
			level[i]+=mnX;
		}
	}

	for(int i=1; i<= nodes; i++){
		vector<int> trail;
		if(doBFS2(i, trail)){
			cpt[i]=trail;
		}
	}

	cout << "******************* <level> **********************\n";
	for(int i=1; i<= nodes; i++){
		cout << level[i] << ",";
	}
	cout <<'\n';

	cout << "******************* <ect> **********************\n";
	for(int i=1; i<= nodes; i++){
		for(int j=1; j<ect[i].size(); j++){
			cout << ect[i][j] << ",";
		}
		cout << '\n';
	}
	
	cout << "****************** <est> ***********************\n";
	for(int i=1; i<= nodes; i++){
		for(int j=1; j<est[i].size(); j++){
			cout << est[i][j] << ",";
		}
		cout << '\n';
	}
	
	cout << "******************** <fproc> *********************\n";
	for(int i=1; i<= nodes; i++){
		for(int j=1; j<fproc[i].size(); j++){
			cout << fproc[i][j] << ",";
		}
		cout << '\n';
	}
	
	cout << "******************** <cpred> *********************\n";
	for(int i=1; i<= nodes; i++){
		cout << cpred[i] << ",";
	}
	cout << '\n';
	
	cout << "******************** <cpred> *********************\n";
	for(int i=1; i<= nodes; i++){
		cout << i << ":";
		for(auto e: cpt[i]){
			cout << e << ",";
		}
		cout << '\n';
	}

	cout << "\ntask clustering:\n";
	task_clustering();
	
	list<int> all_sch[processor+1];
	gen_schedules(all_sch);
	cout<<"makespan: "<<makespan(all_sch)<<'\n';
	verify_sch(all_sch);

	cout << "\ntask duplication:\n";
	task_duplication(all_sch);
	cout<<"makespan: " <<makespan(all_sch)<<'\n';
	verify_sch(all_sch);

	cout << "\nprocessor merging:\n";
	processor_merging(all_sch);
	cout<<"makespan: "<<makespan(all_sch)<<'\n';
	verify_sch(all_sch);
}

void Graph::task_clustering(){

	vector<pii> level_task;
	for(int i=1; i<= nodes; i++){
		level_task.pb(EDGE(level[i], i));
	}
	sort(level_task.begin(), level_task.end());

	entry_task = level_task[nodes-1].se;
	exit_task = level_task[0].se;

	for(auto entry: level_task){
		int task = entry.se;
		// if task assigned to processor: continue, else find first unassigned processor
		if(task_2_processor[task] != -1){
			continue;
		}
		int min_est_proc = -1;
		double min_est_for_task_on_proc = 1000000.0;
		bool found_first_unoccupied_proc = false;
		int proc=1;
		for(; proc<= processor; proc++){
			int fav_proc_of_task = fproc[task][proc];
			if(processor_2_task[fav_proc_of_task] == -1){
				proc = fav_proc_of_task;
				found_first_unoccupied_proc = true;
				break;
			}
			if(min_est_for_task_on_proc > est[task][proc]){
				min_est_for_task_on_proc = est[task][proc];
				min_est_proc = proc;
			}
		}

		if(!found_first_unoccupied_proc){
			task_2_processor[task] = min_est_proc;
			processor_2_task[min_est_proc] = task;
			continue;
		}

		task_2_processor[task] = proc;
		processor_2_task[proc] = task;

		// while it's not entry node
		while(task!=1){
			int criti_pred = cpred[task];
			if(pgraph[task].size()>1 and 
				(task_2_processor[criti_pred]!=-1 or 
					ect[criti_pred][proc] > ect[criti_pred][fproc[criti_pred][1]] +  TT[task][criti_pred]))
			{
				double mN_ect = 1000000.0;
				int found_criti_pred = -1;
				for(auto edge: pgraph[task]){
					int predecessor_task = edge.se;
					if(task_2_processor[predecessor_task] != -1) continue;
					if(ect[predecessor_task][proc] <= 
						ect[predecessor_task][fproc[predecessor_task][1]] + TT[task][predecessor_task]){
						// criti_pred = predecessor_task;
						if(mN_ect > ect[predecessor_task][proc]){
							mN_ect = ect[predecessor_task][proc];
							found_criti_pred = predecessor_task;
						}
					}
				}
				if(found_criti_pred==-1) break;
				else{
					criti_pred = found_criti_pred;
					ect[found_criti_pred][proc] = mN_ect;
				}
			}
			task_2_processor[criti_pred] = proc;
			processor_2_task[proc] = criti_pred;
			task = criti_pred;
		}
		task_2_processor[task] = proc;
		processor_2_task[proc] = task;
	}
	// exit(0);
}

void Graph::update_task2processor(list<int>sch[MAX]){
	for(int p=1; p<= processor; p++){
		for(auto task: sch[p]){
			task_2_processor[task]=p;
		}
	}
}

void Graph::copy_between_start_to_element(list<int>& new_schedule, vector<int> schedule, int task){
	// task should be in the schedule
	auto found_task = find(schedule.begin(), schedule.end(), task);
	assert(found_task!=schedule.end());

	auto start = schedule.begin();
	while(start != found_task){
		new_schedule.pb(*start);
		start++;
	}
}

/* erase 'sch' and  update with 'newSch' entries */
void Graph::update_sch(list<int>* sch, list<int>* newSch){
	for(int proc=1; proc<= processor; proc++){
		sch[proc] = list<int>();
	}
	copy_sch(sch, newSch);
}

void Graph::task_duplication(list<int>* sch){
	list<int> orig_sch[processor+1];
	copy_sch(orig_sch, sch);
	
	int K = 1;
	for(int i=0; i< K; i++){

		// not in use processor
		fill(processor_2_task.begin(), processor_2_task.end(), -1);
		
		// in-use processor set 1		
		for(int w=1; w<=processor; w++){
			if(sch[w].size()>0){
				processor_2_task[w]=1;// processor in use
			}
		}

		// copy curr schedule sch to newSch
		list<int> newSch[processor+1];
		copy_sch(newSch, sch);

		for(int proc=1; proc<= processor; proc++){

			//collect all task on processor "proc"
			list<int> all_task_of_processor = sch[proc];

			if(all_task_of_processor.size()<1) continue;
			
			fill(vis.begin(), vis.end(), false);

			vector<int> topo_order;
			for(auto task : all_task_of_processor){
				if(vis[task]) continue;
				find_curr_allocated_processor(task, sch);
				get_topo_order_of_task_on_processor(proc, task, -1, topo_order, sch);
				topo_order.pb(task);
			}

			// cout << "topological order:\n";
			// for(auto e: topo_order){
			// 	cout << e << ",";
			// }
			// cout << '\n';
			// exit(0);

			vector<int> copied_topo_order(topo_order.size());
			copy(topo_order.begin(), topo_order.end(), copied_topo_order.begin());

			reverse(topo_order.begin(), topo_order.end());
			for(int i=topo_order.size()-1; i>=1; i--){
				int prev_task = topo_order[i-1];
				int curr_task = topo_order[i];
				int cpred_of_task = cpred[curr_task];

				if(prev_task!=cpred_of_task){

					int choosen_processor = -1;
					//find candidate position for task duplication
					for(int fp=1; fp<= processor; fp++){
						int favorite_proc = fproc[prev_task][fp];
						if(processor_2_task[favorite_proc] == -1){
							choosen_processor = favorite_proc;
							break;
						}
					}

					if(choosen_processor == -1){
						choosen_processor = fproc[prev_task][1];
					}

					// move x1 to xi-1 to newProcessor i.e. to choosen processor
					auto it = find(topo_order.begin(), topo_order.end(), curr_task);
					int size = distance(topo_order.begin(), it);
					vector<int> copy_task(size);

					copy(topo_order.begin(), it, copy_task.begin());  

					// moving task to newProcessor
					newSch[choosen_processor] = list<int>();
					for(auto e: copy_task){
						newSch[choosen_processor].pb(e);
					}

					// removing task from curr processor
					set<int> keep_track;

					vector<int>curr_criti_pred_trail = cpt[curr_task];
					for(auto e: curr_criti_pred_trail){
						keep_track.insert(e);
					}
					
					for(auto e: topo_order){
						bool keep = true;
						for(auto p: copy_task){
							if(p == e){
								keep=false; break;
							}
						}
						if(keep){
							keep_track.insert(e);
						}
					}

					list<int>temp(keep_track.size());
					copy(keep_track.begin(), keep_track.end(), temp.begin());

					newSch[proc] = list<int>();
					newSch[proc] = temp;

					// verify_sch(newSch);
					int newMakeSpan =  makespan(newSch);
					int oldMakeSpan = makespan(sch);

					// cout << "\n[1] old makespan:" << oldMakeSpan <<", new makespan:"<< newMakeSpan <<'\n';
					if(newMakeSpan<oldMakeSpan){
						// cout << "\n*************************\n";
						// cout << "new sch\n";
						// verify_sch(newSch);
						// cout << "old sch\n";
						// verify_sch(sch);
						// cout << "\n*************************\n";

						update_sch(sch, newSch);
					}
				}
			}
			
			// if(copied_topo_order[1] != entry_task){
			// 	copy_sch(newSch, sch);

			// 	vector<int>curr_criti_pred_trail = cpt[curr_task];
			// 	set<int> keep_track;

			// 	for(auto e: newSch[proc]){
			// 		curr_criti_pred_trail.pb(e);
			// 	}

			// 	for(auto e: curr_criti_pred_trail){
			// 		keep_track.insert(e);
			// 	}

			// 	list<int> temp(keep_track.size());
			// 	copy(keep_track.begin(), keep_track.end(), temp.begin());
			// 	newSch[proc] = list<int>();
			// 	newSch[proc] = temp;

			// 	int newMakeSpan =  makespan(newSch);
			// 	int oldMakeSpan = makespan(sch);

			// 	cout << "\n[2] old makespan:" << oldMakeSpan <<", new makespan:"<< newMakeSpan <<'\n';
			// 	if(newMakeSpan<oldMakeSpan){
			// 		cout << "\n*************************\n";
			// 		cout << "new sch\n";
			// 		verify_sch(newSch);
			// 		cout << "old sch\n";
			// 		verify_sch(sch);
			// 		cout << "\n*************************\n";
					
			// 		update_sch(sch, newSch);
			// 	}
			// }
		}
	}
}



void Graph::processor_merging(list<int>* sch){
// 1: repeat
// 	2:for every processor p from 1 to m do
// 	3:	Copy the current schedule sch to newSch;
// 	4:	x = the last task on processor p;
// 	5:	Merge tasks on j to processor fprocðx; 1Þ in newSch;
// 	6:	if makespanðnewSchÞ < makespanðschÞ
// 	7:		sch 1⁄4 newSch;
// 	8:end for
// 9:until the above for-loop for K times

	for(int i=1; i<= processor; i++){
		
		int task_count;
		cin>> task_count;
		// cout << "task_count:"<<task_count<<'\n';
		while(task_count){
			int task; cin>>task;
			// cout << "task:"<<task<<'\n';
			sch[i].push_back(task);
			task_count--;
		}
	}

	int k =1;
	list<int>newSch[processor+1];
	copy_sch(newSch, sch);

	while(k--){
		for(int p=1; p<= processor; p++){
			int last_task_on_proc = -1;

			for(auto e: sch[p]){
				last_task_on_proc = e;
			}

			if(last_task_on_proc == -1) continue;
			//fav processor for last task
			// first favourite
			int fav_proc_first = fproc[last_task_on_proc][1];
			
			set<int> temp;
			for(auto task: newSch[p]){
				temp.insert(task);
			}
			for(auto task: newSch[fav_proc_first]){
				temp.insert(task);
			}

			list<int>().swap(newSch[p]);
			list<int>().swap(newSch[fav_proc_first]);
			
			// merge clusters of first and p processor
			for(auto task: temp){
				newSch[fav_proc_first].push_back(task);
			}

			int mk = makespan(sch);
			int new_mk = makespan(newSch);

			cout<< "processor:"<<p<<","<<fav_proc_first<< ", newMakeSpan:"<<new_mk<<", oldMakeSpan:"<<mk<<'\n';
			// cout << "new\n";
			// verify_sch(newSch);
			// cout << "old\n";
			// verify_sch(sch);
			if(new_mk<mk){
				for(int i=1; i<= processor; i++){
					sch[i].clear();
				}
				copy_sch(sch, newSch);
			}
			else{
				for(int i=1; i<= processor; i++){
					newSch[i].clear();
				}
				copy_sch(newSch, sch);
			}
		}
	}
}