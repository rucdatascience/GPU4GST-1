#ifndef __MAPPER__
#define __MAPPER__
#include "header.h"
#include "util.h"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include <thrust/detail/minmax.h>
#include <assert.h>
typedef feature_t (*cb_mapper)(vertex_t active_edge_src,
							   vertex_t active_edge_end,
							   feature_t level,
							   index_t *beg_pos,
							   weight_t weight_list,
							   feature_t *vert_status,
							   feature_t *vert_status_prev);


class mapper
{
public:
	// variable
	vertex_t *adj_list;
	weight_t *weight_list;
	index_t *beg_pos;
	index_t vert_count;
	feature_t *vert_status;
	feature_t *vert_status_prev;
	feature_t *one_label_lower_bound;
	feature_t *lb_record;
	feature_t *merge_or_grow;
	int width, full, *lb0, diameter;
	int val1, val2;
	volatile int *best;
	int *merge_pointer;
	int *merge_groups;
	// index_t *cat_thd_count_sml;
	cb_mapper edge_compute;
	cb_mapper edge_compute_push;
	cb_mapper edge_compute_pull;

public:
	// constructor
	mapper(gpu_graph ggraph, meta_data mdata,
		   cb_mapper user_mapper_push,
		   cb_mapper user_mapper_pull,
		   int wid)
	{
		adj_list = ggraph.adj_list;
		weight_list = ggraph.weight_list;
		beg_pos = ggraph.beg_pos;
		width = wid;
		best = mdata.best;
		full = width - 1;
		diameter = mdata.diameter;
		val1 = width * (diameter + 1), val2 = (diameter + 1);
		merge_groups = ggraph.merge_groups_d;
		merge_pointer = ggraph.merge_pointer_d;


		vert_count = ggraph.vert_count;
		vert_status = mdata.vert_status;
		vert_status_prev = mdata.vert_status_prev;
		// cat_thd_count_sml = mdata.cat_thd_count_sml;

		edge_compute_push = user_mapper_push;
		edge_compute_pull = user_mapper_pull;
	}

	~mapper() {}

public:

	__forceinline__ __device__ void
	mapper_push(		 // 这个是大中小都使用的push函数 这里会进行更新新的距离
		index_t wqueue, // 这个是本轮的用值
		index_t *worklist,
		index_t *cat_thd_count, // 这个是本轮返回值
		const index_t GRP_ID,
		const index_t GRP_SZ,
		const index_t GRP_COUNT,
		const index_t THD_OFF,
		feature_t level)
	{
		index_t appr_work = 0;
		weight_t weight;
		const index_t WSZ = wqueue;
		uint mxid = val1 * (vert_count - 1) + val2 * (width - 1) + diameter;
		for (index_t i = GRP_ID; i < WSZ; i += GRP_COUNT)
		{
			uint frontier = worklist[i];
			int v = frontier / val1;
			int  d = frontier % val2;

			uint p = (frontier - v * val1 - d) / val2;
			index_t beg = beg_pos[v], end = beg_pos[v + 1];
			if (d < diameter)
			{
				for (index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
				{
					// printf("");
					index_t vert_end = adj_list[j];
					weight = weight_list[j];
					index_t update_dest = vert_end * val1 + p * val2  +d+ 1;
					feature_t dist = vert_status[frontier]+weight; // 获取值 根据任务选择操作
					if (vert_status[update_dest] > dist)
					{
						
						atomicMin(vert_status + update_dest, dist);
					}
				}
			}

			// merge
			beg = merge_pointer[p], end = merge_pointer[p + 1];
			int dist;
			for (index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
			{	
				uint p_comp = merge_groups[j];
				for (uint dia = 0; dia <= diameter - d; dia++)
				{
	
					weight_t weight = vert_status[v * val1 + p_comp * val2 + dia];
					uint new_d = dia>d?dia:d;
					
					index_t update_dest = v * val1 + (p_comp | p) * val2 + new_d;

					if (frontier > mxid) {
						printf("123");
					}
					dist = vert_status[frontier] + weight;

					if (vert_status[update_dest] > dist)
					{
						
						atomicMin(vert_status + update_dest, dist);
					}
				}
			}
		}


		cat_thd_count[threadIdx.x + blockIdx.x * blockDim.x] = appr_work;
	}

	

	


};

#endif
