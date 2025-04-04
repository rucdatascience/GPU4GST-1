#ifndef __REDUCER_H__
#define __REDUCER_H__

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include "meta_data.cuh"
#include "gpu_graph.cuh"
#include <limits.h>
#include <assert.h>

__global__ void 
thread_stride_gather(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		reducer reducer_inst
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	//const index_t GRNTY = blockDim.x * gridDim.x;
	//const index_t WOFF = threadIdx.x & 31;
	//const index_t wid_in_blk = threadIdx.x >> 5;
	//const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t BIN_OFF = TID * BIN_SZ;

	//worklist_gather._thread_stride_gather
	//		(mdata.worklist_mid, 
	//		 mdata.worklist_bin, 
	//		 my_front_count, 
	//		 output_off, 
	//		 BIN_OFF);

	reducer_inst._thread_stride_gather(
	mdata.worklist_mid, mdata.worklist_bin, mdata.cat_thd_count_mid[TID],
    mdata.cat_thd_off_mid[TID], BIN_OFF);
}


/* Scan status array to generate *sorted* frontier queue   不是严格有序而是分成大中小三段*/
__global__ void 
gen_push_worklist(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		reducer reducer_inst
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t WOFF = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;

	reducer_inst._push_coalesced_scan_random_list(
	TID, wid_in_blk, WOFF, wcount_in_blk, GRNTY,	level);
}






/* Call by users */
int reducer_push
(feature_t level,
 gpu_graph ggraph,
 meta_data mdata,
 reducer worklist_gather
 ){
	gen_push_worklist<<<BLKS_NUM, THDS_NUM>>>
		(level, ggraph, mdata, worklist_gather);
	H_ERR(cudaDeviceSynchronize());
	return 0;
}

// 假设有 G 个 group，每个 group 用一个二进制位表示
// 例如 G=3 时：
// group 0: 001 (1 << 0)
// group 1: 010 (1 << 1)
// group 2: 100 (1 << 2)
// 组合 0+1: 011 (3)
// 组合 1+2: 110 (6)
// 组合 0+1+2: 111 (7)

// 预计算掩码
__constant__ feature_t group_masks[MAX_GROUPS];  // 存储每个 group 的掩码
__constant__ feature_t valid_combinations[1 << MAX_GROUPS];  // 存储有效的组合掩码

// 在初始化时预计算所有可能的掩码
void init_group_masks(int G) {
    // 预计算单个 group 的掩码
    feature_t *h_masks = new feature_t[G];
    for (int i = 0; i < G; i++) {
        h_masks[i] = 1 << i;  // 每个 group 对应一个二进制位
    }
    cudaMemcpyToSymbol(group_masks, h_masks, sizeof(feature_t) * G);
    
    // 预计算所有可能的组合掩码
    feature_t *h_combinations = new feature_t[1 << G];
    for (int i = 0; i < (1 << G); i++) {
        h_combinations[i] = i;  // 每个组合对应一个唯一的掩码
    }
    cudaMemcpyToSymbol(valid_combinations, h_combinations, sizeof(feature_t) * (1 << G));
    
    delete[] h_masks;
    delete[] h_combinations;
}

__global__ void init_query_masks(
    vertex_t *queries,
    int num_queries,
    feature_t *vert_status,
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    vertex_t query = queries[idx];
    int v = query / width;
    int p = query % width;
    
    // 使用预计算的掩码快速设置状态
    for (int i = 0; i < width; i++) {
        if (valid_combinations[i]) {
            vertex_t update_dest = v * width + i;
            vert_status[update_dest] = 0;  // 设置初始状态
        }
    }
}

__device__ void merge_with_mask(
    vertex_t v,
    feature_t p,
    feature_t *vert_status,
    int width)
{
    int vline = v * width;
    feature_t current_mask = p;
    
    // 使用预计算的掩码快速找到需要合并的组合
    for (int i = 0; i < width; i++) {
        if ((i & current_mask) == 0) {  // 检查是否可以合并
            feature_t new_mask = i | current_mask;
            vertex_t update_dest = vline + new_mask;
            
            // 计算新的距离
            feature_t dist = vert_status[vline + i] + vert_status[vline + current_mask];
            if (vert_status[update_dest] > dist) {
                atomicMin(vert_status + update_dest, dist);
            }
        }
    }
}

__device__ feature_t compute_lower_bound_with_mask(
    vertex_t v,
    feature_t p,
    feature_t *vert_status,
    int width)
{
    int vline = v * width;
    feature_t complement = (width - 1) ^ p;  // 计算补集掩码
    feature_t lb = 0;
    
    // 使用预计算的掩码快速计算下界
    for (int i = 0; i < width; i++) {
        if ((i & complement) == i) {  // 检查是否是补集的子集
            lb = max(lb, vert_status[vline + i]);
        }
    }
    
    return lb;
}

__global__ void process_queries_with_shared_memory(
    vertex_t *queries,
    int num_queries,
    feature_t *vert_status,
    int width)
{
    __shared__ feature_t shared_masks[THREAD_PER_BLOCK];
    __shared__ feature_t shared_combinations[THREAD_PER_BLOCK];
    
    // 加载掩码到共享内存
    if (threadIdx.x < width) {
        shared_masks[threadIdx.x] = group_masks[threadIdx.x];
        shared_combinations[threadIdx.x] = valid_combinations[threadIdx.x];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    // 使用共享内存中的掩码处理查询
    vertex_t query = queries[idx];
    int v = query / width;
    int p = query % width;
    
    for (int i = 0; i < width; i++) {
        if (shared_combinations[i]) {
            vertex_t update_dest = v * width + i;
            vert_status[update_dest] = 0;
        }
    }
}

__global__ void analyze_graph_structure(
    vertex_t vert_count,
    index_t edge_count,
    index_t *csr_offset,
    vertex_t *csr_edges,
    meta_data::NodeType *node_types,
    int *community_sizes,
    float *node_importance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vert_count) return;
    
    // 计算节点度
    int degree = csr_offset[idx + 1] - csr_offset[idx];
    
    // 识别节点类型
    meta_data::NodeType type;
    type.is_hub = (degree > 1000);  // 高阈值表示枢纽节点
    type.is_leaf = (degree == 1);   // 度为1表示叶子节点
    
    // 计算局部中心性
    float centrality = 0.0f;
    for (int i = csr_offset[idx]; i < csr_offset[idx + 1]; i++) {
        vertex_t neighbor = csr_edges[i];
        int neighbor_degree = csr_offset[neighbor + 1] - csr_offset[neighbor];
        centrality += 1.0f / (neighbor_degree + 1);
    }
    
    // 识别桥接节点
    type.is_bridge = false;
    if (!type.is_hub && !type.is_leaf) {
        int bridge_count = 0;
        for (int i = csr_offset[idx]; i < csr_offset[idx + 1]; i++) {
            vertex_t neighbor = csr_edges[i];
            if (node_types[neighbor].is_hub) {
                bridge_count++;
            }
        }
        type.is_bridge = (bridge_count >= 2);
    }
    
    // 计算社区ID（使用简单的启发式方法）
    int community_id = idx;
    if (type.is_hub) {
        community_id = -1;  // 特殊社区ID表示枢纽节点
    } else if (type.is_leaf) {
        community_id = csr_edges[csr_offset[idx]];  // 叶子节点加入其邻居的社区
    }
    
    // 保存结果
    node_types[idx] = type;
    community_sizes[idx] = (community_id == -1) ? 1 : 0;
    node_importance[idx] = centrality;
}

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
    
    // 根据节点类型优化处理策略
    meta_data::NodeType src_type = node_types[src];
    meta_data::NodeType dest_type = node_types[update_dest];
    
    // 对枢纽节点使用特殊处理
    if (src_type.is_hub || dest_type.is_hub) {
        // 使用更激进的剪枝策略
        if (vert_status[update_dest] > dist + node_importance[update_dest]) {
            atomicMin(vert_status + update_dest, dist);
            // ... 更新工作队列 ...
        }
    }
    // 对叶子节点使用简化处理
    else if (src_type.is_leaf || dest_type.is_leaf) {
        if (vert_status[update_dest] > dist) {
            atomicMin(vert_status + update_dest, dist);
            // ... 更新工作队列 ...
        }
    }
    // 对桥接节点使用保守处理
    else if (src_type.is_bridge || dest_type.is_bridge) {
        if (vert_status[update_dest] > dist + 0.5f * node_importance[update_dest]) {
            atomicMin(vert_status + update_dest, dist);
            // ... 更新工作队列 ...
        }
    }
    // 普通节点使用标准处理
    else {
        if (vert_status[update_dest] > dist) {
            atomicMin(vert_status + update_dest, dist);
            // ... 更新工作队列 ...
        }
    }
    
    // ... 其他代码 ...
}

__global__ void assign_to_work_queues(
    vertex_t *worklist,
    vertex_t work_size,
    meta_data::NodeType *node_types,
    int *community_sizes,
    vertex_t *queue1,
    vertex_t *queue2,
    vertex_t *queue3,
    vertex_t *queue_sizes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= work_size) return;
    
    vertex_t v = worklist[idx];
    meta_data::NodeType type = node_types[v];
    
    // 根据节点类型和社区信息分配队列
    if (type.is_hub) {
        // 枢纽节点分配到队列1
        int pos = atomicAdd(queue_sizes, 1);
        queue1[pos] = v;
    }
    else if (type.is_leaf || community_sizes[v] < 10) {
        // 叶子节点和小社区节点分配到队列2
        int pos = atomicAdd(queue_sizes + 1, 1);
        queue2[pos] = v;
    }
    else {
        // 其他节点分配到队列3
        int pos = atomicAdd(queue_sizes + 2, 1);
        queue3[pos] = v;
    }
}

#endif
