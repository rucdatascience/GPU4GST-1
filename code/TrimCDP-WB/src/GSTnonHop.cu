#include "header.h"
#include "util.h"
#include "mapper.cuh"
#include "reducer.cuh"
#include "wtime.h"
#include "barrier.cuh"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "mapper_enactor.cuh"
#include "reducer_enactor.cuh"
#include <thrust/detail/minmax.h>
#include <thrust/fill.h>
#define THREAD_PER_BLOCK 512
using namespace std;
/*user defined vertex behavior function*/

__inline__ __host__ __device__ feature_t user_mapper_push // reduce edge_compute_push
	(vertex_t src,
	 vertex_t dest,
	 feature_t level,
	 index_t *beg_pos,
	 weight_t edge_weight,
	 feature_t *vert_status,
	 feature_t *vert_status_prev)
{
	// if (vert_status_prev[src] == vert_status[src])
	return vert_status[src] + edge_weight;
	// else
	//     return INFTY;
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_push(vertex_t vert_id,
														 feature_t level,
														 vertex_t *adj_list,
														 index_t *beg_pos,
														 feature_t *vert_status,
														 feature_t *vert_status_prev)
{
	return (vert_status[vert_id] != vert_status_prev[vert_id]);
}
__inline__ __device__ bool vertex_selector_best_push(vertex_t vert_id,
													 vertex_t width,
													 feature_t *vert_status,
													 feature_t *vert_status_prev,
													 feature_t *one_label,
													 volatile feature_t *best,
													 feature_t *lb_record,
													 feature_t *merge_or_grow,
													 feature_t *temp_store)
{
	if (vert_status[vert_id] != vert_status_prev[vert_id] && vert_status[vert_id] - 1 <= 0.5 * (*best))
	{

		if (lb_record[vert_id] + vert_status[vert_id] <= (*best))
			return true;
	}

	return false;
}
/*user defined vertex behavior function*/
__inline__ __host__ __device__ feature_t user_mapper_pull(vertex_t src,
														  vertex_t dest,
														  feature_t level,
														  index_t *beg_pos,
														  weight_t edge_weight,
														  feature_t *vert_status,
														  feature_t *vert_status_prev)
{
	// return vert_status[src] + edge_weight;
	return vert_status[src] + edge_weight;
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_pull // reduce
	(vertex_t vert_id,
	 feature_t level,
	 vertex_t *adj_list,
	 index_t *beg_pos,
	 feature_t *vert_status,
	 feature_t *vert_status_prev)
{
	return (beg_pos[vert_id] != beg_pos[vert_id + 1]);
	// return (vert_status[vert_id] != vert_status_prev[vert_id]);
	// return (vert_status[vert_id] != vert_status_prev[vert_id] || vert_status[vert_id] == INFTY);
	// return true;
}
__global__ void one_label_lb_1(int *tree, int width, int *lb0, int N, int G, int *can_find_solution, int *w, int *w1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		int row = idx * width, temp = 0;

		for (size_t i = 1; i < width; i <<= 1)
		{
			if (temp < tree[row + i])
			{
				temp = tree[row + i];
			}
		}
		if (temp != inf)
		{
			*can_find_solution = 1;
		}
		int vline = idx * width;
		for (int x = 1; x < width; x++)
		{
			lb0[vline + x] = inf;
			for (int i = 0; i < G; i++)
			{
				for (int j = 0; j < G; j++)
				{
					lb0[vline + x] = thrust::min(lb0[vline + x], tree[vline + (1 << i)] + w[i * width * G + j * width + x] + tree[vline + (1 << j)]);
				}
			}
			int minj = inf;
			for (size_t i = 0; i < G; i++)
			{
				if ((1 << i) & x)
				{
					minj = thrust::min(minj, tree[idx * width + (1 << i)]);
				}
			}
			for (size_t i = 0; i < G; i++)
			{
				int vi = 1 << i;
				if (vi & x)
				{
					lb0[vline + x] = thrust::max(lb0[vline + x], tree[vline + (1 << i)] + w1[i * width + x] + minj);
				}
			}

			lb0[vline + x] /= 2;
		}
	}
}

__global__ void count_used(int *tree, int width, int *counts,int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		int vline = idx * width;
		for (int x = 1; x < width; x++)
		{
			if(tree[vline+x]!=inf)
			atomicAdd(counts,1);
			
		}
	}
}

__device__ cb_reducer vert_selector_push_d = vertex_selector_push;
__device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
__device__ cb_mapper vert_behave_push_d = user_mapper_push;
__device__ cb_mapper vert_behave_pull_d = user_mapper_pull;
__device__ best_reducer vertex_selector_best_push_d = vertex_selector_best_push;

__forceinline__ __device__ void
mapper_push(vertex_t wqueue,
           vertex_t *worklist,
           index_t *cat_thd_count,
           const index_t GRP_ID,
           const index_t GRP_SZ,
           const index_t GRP_COUNT,
           const index_t THD_OFF,
           feature_t level,
           volatile vertex_t *bests,
           feature_t *records)
{
    // ... 现有的代码 ...
    
    // 在处理完顶点后，将新生成的顶点分配到桶中
    if (vert_status[update_dest] > dist) {
        atomicMin(vert_status + update_dest, dist);
        
        // 根据新的距离将顶点分配到桶中
        int bucket_idx = 0;
        while (bucket_idx < NUM_BUCKETS && dist >= bucket_ranges[bucket_idx]) {
            bucket_idx++;
        }
        if (bucket_idx < NUM_BUCKETS) {
            int pos = atomicAdd(bucket_sizes[bucket_idx], 1);
            buckets[bucket_idx][pos] = update_dest;
        }
    }
    
    // ... 其他代码 ...
}

int main(int args, char **argv)
{
	string path = argv[2], data_name = argv[3];
	int T = atoi(argv[4]), task_start_num = atoi(argv[5]), task_end_num = atoi(argv[6]);
	// string path = "/home/lijiayu/simd/data/", data_name = "com-amazon";
	// int T = 4, task_start_num =0, task_end_num = 1999;
	// double tm_map, tm_red, tm_scan;   无直径约束gpu2
	string file_beg = path + data_name + "_beg_pos.bin";
	string file_adj = path + data_name + "_csr.bin";
	string file_weight = path + data_name + "_weight.bin";

	const char *file_beg_pos = file_beg.c_str();
	const char *file_adj_list = file_adj.c_str();
	const char *file_weight_list = file_weight.c_str();
	int blk_size = 512;
	// int switch_iter = 100;
	int *non_overlapped_group_gpu, *non_overlapped_group_pointer, *can_find, *w_d, *w1_d;
	// Read graph to CPU
	graph<long, long, long, vertex_t, index_t, weight_t>
	*ginst = new graph<long, long, long, vertex_t, index_t, weight_t>(file_beg_pos, file_adj_list, file_weight_list);
	ginst->read_gi(path + data_name + ".g", path + data_name + to_string(T) + ".csv");
	cout<<"inquire have "<<ginst->inquire.size()<<endl;
	int G = ginst->inquire[0].size(), V = ginst->vert_count;
	int width = 1 << G, problem_size = V * width, sum_cost = 0;
	non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(width - 1);
	cudaMallocManaged((void **)&non_overlapped_group_gpu, sizeof(int) * s.length);
	cudaMallocManaged((void **)&non_overlapped_group_pointer, sizeof(int) * (width + 1));
	cudaMallocManaged((void **)&w_d, G * G * width * sizeof(int));
	cudaMallocManaged((void **)&w1_d, G * width * sizeof(int));
	cudaMemcpy(non_overlapped_group_gpu, s.non_overlapped_group_sets_IDs.data(), sizeof(int) * s.length, cudaMemcpyHostToDevice);
	cudaMemcpy(non_overlapped_group_pointer, s.non_overlapped_group_sets_IDs_pointer_host.data(), (width + 1) * sizeof(int), cudaMemcpyHostToDevice);

	meta_data mdata(ginst->vert_count, ginst->edge_count, width);
	std::vector<int> contain_group_vertices, costs(100);
	int *host_tree = new int[width * V], *inqueue = new int[problem_size];int *one_label_h = new int[width * V];
	int w[G * G * width], w1[G * width], vv[G][G];
	std::fill(inqueue, inqueue + problem_size, 0);
	feature_t *level, *level_h;
	cudaMalloc((void **)&level, 10 * sizeof(feature_t));
	cudaMallocHost((void **)&level_h, 10 * sizeof(feature_t));
	cudaMemset(level, 0, sizeof(feature_t));
	cudaMallocManaged((void **)&can_find, sizeof(int));
	cb_reducer vert_selector_push_h;
	cb_reducer vert_selector_pull_h;
	best_reducer vert_selector_push_h_best;
	cudaMemcpyFromSymbol(&vert_selector_push_h_best, vertex_selector_best_push_d, sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_selector_push_h, vert_selector_push_d, sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_selector_pull_h, vert_selector_pull_d, sizeof(cb_reducer));

	cb_mapper vert_behave_push_h;
	cb_mapper vert_behave_pull_h;

	cudaMemcpyFromSymbol(&vert_behave_push_h, vert_behave_push_d, sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_behave_pull_h, vert_behave_pull_d, sizeof(cb_reducer));
	// Init three data structures
	gpu_graph ggraph(ginst);

	ggraph.merge_groups_d = non_overlapped_group_gpu, ggraph.merge_pointer_d = non_overlapped_group_pointer;
	double sum_time = 0;
	cout << "width " << mdata.width << " V " << ginst->vert_count << " E " << ginst->edge_count << endl;
	for (int ii = 0; ii < ginst->inquire.size(); ii++)
	{
		for (int jj = 0; jj < ginst->inquire[ii].size(); jj++)
		{
			// ginst->inquire[ii][jj] -= (V);
		}
	}
	ofstream outputFile;
	outputFile.precision(8);
	outputFile.setf(ios::fixed);
	outputFile.setf(ios::showpoint);
	outputFile.open(path+"result/exp_GPU2_nonHop_" + data_name + "_T" + to_string(T) + "_" + to_string(task_start_num) + "-" + to_string(task_end_num) + ".csv");

	outputFile << "task_ID,task,GPU2_nonHop_time,GPU2_nonHop_cost,GPU2_nonHop_memory,counts,process_num" << endl;
	Barrier global_barrier(BLKS_NUM);
	mapper compute_mapper(ggraph, mdata, vert_behave_push_h, vert_behave_pull_h, width);
	reducer worklist_gather(ggraph, mdata, vert_selector_push_h, vert_selector_pull_h, vert_selector_push_h_best, width);

	// 初始化桶
	mdata.init_buckets(ginst->vert_count, width);
	
	for (int ii = task_start_num; ii <= task_end_num; ii++)
	{
		string task = "";
		for (size_t j = 0; j < ginst->inquire[ii].size(); j++)
		{
			task += to_string(ginst->inquire[ii][j]) + " ";
		}
		std::cout << "------------------------------------------------------------" << endl;
		cout <<data_name  <<" interation " << ii << " "<<endl;
		//  cout << ginst->inquire[i][j] << " ";
		std::fill(host_tree, host_tree + width * V, inf);
		cudaMemcpy(mdata.vert_status_prev, host_tree, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		contain_group_vertices.clear();
		set_max_ID(ginst->group_graph, ginst->inquire[ii], host_tree, contain_group_vertices, width); // 在置初值前传递参数到prev
		for (auto it = contain_group_vertices.begin(); it != contain_group_vertices.end(); it++)
		{
			int v = *it;
			int group_set_ID_v = get_max(v, host_tree, width); /*time complexity: O(|Gamma|)*/
			for (size_t j = 1; j <= group_set_ID_v; j <<= 1)
			{
				if (j & group_set_ID_v)
				{
					host_tree[v * width + j] = 0;
				}
			}
		}
		cudaMemcpy(mdata.vert_status, host_tree, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//* necessary for high diameter graph, e.g., euro.osm and roadnet.ca
		// mapper_merge_push(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
		double stime = wtime();
		cudaMemset(level, 0, 10 * sizeof(feature_t));
		
		mapper_hybrid_push_merge(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier, 1);
		double sstime = wtime() - stime;
		H_ERR(cudaMemcpy(mdata.record, mdata.vert_status, problem_size * sizeof(int), cudaMemcpyDeviceToDevice));

		// for (size_t i = 0; i < V; i++)
		// {
		// 	cout << " v " << mdata.worklist_mid[i] / width << " p " << mdata.worklist_mid[i] % width << "  ";
		// }
		// cout << endl;
		std::fill(host_tree, host_tree + problem_size, inf);
		cudaMemcpy(mdata.vert_status_prev, host_tree, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		
		std::fill(inqueue, inqueue + problem_size, 0);
		set_max_ID(ginst->group_graph, ginst->inquire[ii], host_tree, contain_group_vertices, width); // 在置初值前传递参数到prev
		for (auto it = contain_group_vertices.begin(); it != contain_group_vertices.end(); it++)
		{
			int v = *it;
			int group_set_ID_v = get_max(v, host_tree, width); /*time complexity: O(|Gamma|)*/
			for (size_t i = 1; i <= group_set_ID_v; i++)
			{
				if ((i | group_set_ID_v) == group_set_ID_v)
				{
					host_tree[v * width + i] = 0;
					inqueue[v * width + i] = 1;
				}
			}
		}
		cudaMemcpy(mdata.merge_or_grow, inqueue, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		// for (size_t i = 0; i < *dis_queue_size; i++)
		// {
		// 	cout << " v " << dis_queue[i] / width << " p " << dis_queue[i] % width << "  ";
		// }
		// cout << endl;
		
		for (size_t i = 0; i < G; i++)
		{
			for (size_t j = 0; j < G; j++)
			{
				vv[i][j] = inf;
				for (size_t k = 0; k < width; k++)
				{
					w[i * width * G + j * width + k] = inf;
				}
			}
		}
		cudaMemcpy(mdata.vert_status, host_tree, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		// cudaMemcpy(&work_size, (void **)&mdata.worklist_sz_sml, sizeof(int), cudaMemcpyDeviceToHost);
		// cout << "work size " << work_size << endl;
		// mapper_hybrid_push_merge(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier, 0);
		
		cudaMemcpy(one_label_h, mdata.record, width * V * sizeof(int), cudaMemcpyDeviceToHost);
		for (size_t i = 0; i < 5; i++)
		{
			// cout << i << "  ";
			for (size_t j = 1; j < width; j <<= 1)
			{
				//	cout << one_label_h[i * width + j] << " ";
			}
			// cout << endl;
		}
		for (size_t i = 0; i < V; i++)
		{
			for (size_t j = 0; j < G; j++)
			{
				if (one_label_h[i * width + (1 << j)] == 0)
				{
					for (size_t k = 0; k < G; k++)
					{
						vv[j][k] = min(vv[j][k], one_label_h[i * width + (1 << k)]);
					}
				}
			}
		}
		for (size_t i = 0; i < G; i++)
		{
			// cout << i << " ";
			w[i * width * G + i * width + (1 << i)] = 0;
			for (size_t j = 0; j < G; j++)
			{
				//	cout << vv[i][j] << " ";
				w[i * width * G + j * width] = vv[i][j];
			}
			// cout << endl;
		}
		for (int x = 1; x < width; x++)
		{
			for (int i = 0; i < G; i++)
			{
				for (int j = 0; j < G; j++)
				{
					int xx = x;
					if (xx & (1 << j))
						xx -= (1 << j);
					if (xx & (1 << i))
						xx -= (1 << i);
					if (xx == 0)
					{
						w[i * width * G + j * width + x] = w[i * width * G + j * width];
					}
					for (int p = 0; p < G; p++)
					{
						if ((1 << p) & (xx)) // p作为中间节点的首要条件是p要是xx的一个点
						{
							// printf("x=%d i=%d j=%d mid=%d ,compare %d %d\n", x, i, j, 1 << p, w[i][j][x], w[i][p][xx - (1 << p)] + vv[p][j]);

							w[i * width * G + j * width + x] = min(w[i * width * G + j * width + x], w[i * width * G + p * width + xx - (1 << p)] + vv[p][j]);
						}
					}
					// printf("set  i=%d j=%d, x=%d W=%d\n", i, j, x, w[i][j][x]);
				}
			}
		}

		for (int i = 0; i < G; i++)
		{
			for (size_t x = 1; x < width; x++)
			{
				w1[i * width + x] = inf;
				for (size_t j = 0; j < G; j++)
				{
					w1[i * width + x] = min(w1[i * width + x], w[i * width * G + j * width + x]);
				}
			}
		}
		cudaMemcpy((void **)&w_d, w, G * G * width * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy((void **)&w1_d, w1, G * width * sizeof(int), cudaMemcpyHostToDevice);
		*can_find = 0;
		one_label_lb_1<<<(V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(mdata.record, width, mdata.lb0, V, G, can_find, w_d, w1_d);
		cudaDeviceSynchronize();
		if (*can_find == 0)
		{
			cout << "can not find a solution" << endl;
			//	return solution_tree;
		}

		double time = wtime();
		// mapper_hybrid_push_merge(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier, 0);

		// 处理工作队列
		int total_processed = 0;
		while (true) {
			// 从桶中收集工作队列
			collect_from_buckets<<<(target_size + 255) / 256, 256>>>(
				mdata.worklist_mid,
				mdata,
				target_size);
			
			// 处理工作队列
			mapper_push(...);
			
			// 更新桶的范围
			update_bucket_ranges<<<1, mdata.NUM_BUCKETS>>>(
				mdata,
				level,
				total_processed);
			
			// 将新生成的顶点分配到桶中
			distribute_to_buckets<<<(work_size + 255) / 256, 256>>>(
				mdata.worklist_mid,
				work_size,
				mdata.vert_status,
				mdata);
			
			total_processed += work_size;
			
			// 检查是否完成
			if (work_size == 0) break;
		}
		
		double ftime = wtime() - time; // 一阶段推的耗时
		cudaMemcpy(level_h, level, 10 * sizeof(feature_t), cudaMemcpyDeviceToHost);
		//std::cout << "iteration: " << level_h[0] << " queue size " << level_h[1] << " future work " << level_h[2] << "overflow " << level_h[3]<<" used "<<level_h[5] << "\n";
		cudaMemcpy(host_tree, mdata.vert_status, problem_size * sizeof(int), cudaMemcpyDeviceToHost);
		// for (size_t i = 0; i < V; i++)
		// {
		// 	cout << i << "  ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		cout << host_tree[i * width + j] << " ";
		// 	}
		// 	cout << endl;
		// }
		//cudaMemcpy(one_label_h, worklist_gather.temp_st, width * V * sizeof(int), cudaMemcpyDeviceToHost);
		// 			for (size_t i = 0; i <V; i++)
		// {
		// 	cout << i << "  ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		cout << one_label_h[i * width + j] << " ";
		// 	}
		// 	cout << endl;
		// }
		int minc = inf;
		for (size_t i = 0; i < V; i++)
		{
			minc = min(minc, host_tree[i * width + width - 1]);
		}
		cout << "min cost " << minc << endl;
		sum_cost += minc;
		// balanced_push(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);

	
		// cudaMemcpy(inqueue, mdata.merge_or_grow, problem_size * sizeof(int), cudaMemcpyDeviceToHost);
		// cudaMemcpy(host_tree, mdata.vert_status, problem_size * sizeof(int), cudaMemcpyDeviceToHost);

		// cudaMemcpy(work_list, mdata.new_worklist_sml, problem_size * sizeof(int), cudaMemcpyDeviceToHost);

		// cudaMemcpy(&work_size, (void **)&mdata.new_worklist_sz_sml, sizeof(int), cudaMemcpyDeviceToHost);
		// cout << "work size " << work_size << endl;
		// for (size_t i = 0; i < V; i++)
		// {
		// 	cout << i << "  ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		cout << host_tree[i * width + j] << " ";
		// 	}
		// 	cout << endl;
		// }
		// for (size_t i = 0; i < V; i++)
		// {
		// 	cout << i << "  ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		cout << inqueue[i * width + j] << " ";
		// 	}
		// 	cout << endl;
		// }

		// for (size_t i = 0; i < problem_size; i++)
		// {
		// 	cout << work_list[i] / width << "-" << work_list[i] % width << " ";
		// }
		// cout << endl;
		//* works for low diameter graphs.
		// push_pull_opt(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
		time = wtime() - time;
		int *set_num;
		cudaMallocManaged((void **)&set_num,sizeof(int));

		count_used<<<(V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(mdata.vert_status, width, set_num, V);
		cudaDeviceSynchronize();
		int mem_MB = (level_h[5]+V*width+*set_num);
		//cout<<"set num "<<*set_num<<" queue size "<<level_h[5]<<" V*width "<<V*width<<endl;
		outputFile << ii << "," << task << "," << ftime << "," << minc << "," << mem_MB<<","<<*set_num<<","<<level_h[6] << endl;
		sum_time += ftime;
		// std::cout << "Total iteration: " << level_h[0] << "\n";
	}
	delete[] host_tree;delete []inqueue;delete []one_label_h;

	mdata.release();
	ggraph.release();
	cudaFree(w1_d);
	cudaFree(w_d);
	cudaFree(non_overlapped_group_gpu);
	cudaFree(non_overlapped_group_pointer);
	return 0;
	// delete[] gpu_dist;
}
