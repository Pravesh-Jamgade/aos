
			// for all task where "cpred of current task" is NOT EQUAL "prev task"
			for(int k=tasks.size()-1; k>= 2; k++){

				// find candidate for task-duplication
				if(tasks[k-1] != cpred[tasks[k]]){

					// from all fproc of task k, find first unoccupied processor
					bool found_first_unoccupied = false;

					int fp=1;
					for(; fp<= processor; fp++){
						// found first unoccupied processor
						if(processor_2_task[fproc[k][fp]] == -1){
							found_first_unoccupied = true;
							break;
						}
					}
					
					int next_proc;
					if(found_first_unoccupied) next_proc = fp;
					else next_proc = fproc[k][1];

					//find task 'k' processor
					auto it_task = find(copied_tasks.begin(), copied_tasks.end(), tasks[k]);

					// copy current schedule
					list<int>* curr_sch;
					copy_sch(curr_sch, sch);

					// find current makespan
					int curr_makespan = makespan(curr_sch);

					// if so, move x1 to xi-1 on (new)next_proc
					list<int> newsch(tasks.size());
					copy_between_start_to_element(newsch, copied_tasks, tasks[k]);
					sch[next_proc]=newsch;

					// remove from current processor schedule
					for(auto entry: newsch){
						remove(sch[p].begin(), sch[p].end(), entry);
					}

					// add cpt of tasks[k] to processor p
					vector<int> cpt_of_p = cpt[p];

					// append existing task of p, to cpt of p
					for(auto entry: sch[p]){
						cpt_of_p.pb(entry);
					}

					// remove remaning schedule of p
					sch[p] = list<int>(tasks.size());

					// add new schedule to p
					copy(cpt_of_p.begin(), cpt_of_p.end(), sch[p].begin());

					// find new makespan
					int new_makespan = makespan(sch);

					cout << "[1] old makespan:" << curr_makespan << "," << " new makespan:" << new_makespan << '\n';
					// accept new schedule if..else keep old
					if(new_makespan < curr_makespan){

					}
					else{
						sch = curr_sch;
					}
				}
			}

			if(tasks[0] != entry_task){

					// copy current schedule
					list<int>* curr_sch;
					copy_sch(curr_sch, sch);

					// find current makespan
					int curr_makespan = makespan(curr_sch);

					// add cpt of tasks[k] to processor p
					vector<int> cpt_of_p = cpt[p];

					// append existing task of p, to cpt of p
					for(auto entry: sch[p]){
						cpt_of_p.pb(entry);
					}

					// remove remaning schedule of p
					sch[p] = list<int>(tasks.size());

					// add new schedule to p
					copy(cpt_of_p.begin(), cpt_of_p.end(), sch[p].begin());

					// find new makespan
					int new_makespan = makespan(sch);

					cout << "[2] old makespan:" << curr_makespan << "," << " new makespan:" << new_makespan << '\n';
					// accept new schedule if..else keep old
					if(new_makespan < curr_makespan){

					}
					else{
						sch = curr_sch;
					}
			}
