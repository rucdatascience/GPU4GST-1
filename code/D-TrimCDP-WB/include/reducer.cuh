#ifndef __REDUCER__
#define __REDUCER__

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include "meta_data.cuh"
#include "gpu_graph.cuh"
#include <limits.h>
#include <assert.h>

/*User provided virtual function*/
typedef bool (*cb_reducer)(vertex_t, feature_t, vertex_t *, index_t *,
						   feature_t *, feature_t *);
/* Gather data from bin to global worklist: thread-strided manner*/
// template<typename vertex_t, typename index_t>
class reducer
{
public:
	// variables
	vertex_t *adj_list;
	index_t *beg_pos;
	index_t vert_count;
	feature_t *vert_status;
	feature_t *vert_status_prev;
	feature_t *one_label_lower_bound;
	feature_t *one_label_h;
	index_t *worklist_sml;
	index_t *worklist_mid;
	index_t *worklist_lrg;
	int width, diameter;
	volatile vertex_t *worklist_sz_sml;
	volatile vertex_t *worklist_sz_mid;
	volatile vertex_t *worklist_sz_lrg;
	int VAL1, VAL2;
	index_t *cat_thd_count_sml;
	index_t *cat_thd_count_mid;
	index_t *cat_thd_count_lrg;

	cb_reducer vert_selector_push;
	cb_reducer vert_selector_pull;

public:
	// constructor
	reducer(gpu_graph ggraph,
			meta_data mdata,
			cb_reducer user_reducer_push,
			cb_reducer user_reducer_pull,
			int wid)
	{
		adj_list = ggraph.adj_list;
		beg_pos = ggraph.beg_pos;
		vert_count = ggraph.vert_count;
		width = wid;
		diameter = mdata.diameter;
		VAL1 = width * (diameter + 1), VAL2 = (diameter + 1);
	
		vert_status = mdata.vert_status;
		vert_status_prev = mdata.vert_status_prev;
		worklist_sml = mdata.worklist_sml;
		worklist_mid = mdata.worklist_mid;
		worklist_lrg = mdata.worklist_lrg;
		worklist_sz_sml = mdata.worklist_sz_sml;
		worklist_sz_mid = mdata.worklist_sz_mid;
		worklist_sz_lrg = mdata.worklist_sz_lrg;

		vert_selector_push = user_reducer_push;
		vert_selector_pull = user_reducer_pull;
		cat_thd_count_sml = mdata.cat_thd_count_sml;
		cat_thd_count_mid = mdata.cat_thd_count_mid;
		cat_thd_count_lrg = mdata.cat_thd_count_lrg;
	}

public:
	// functions
	/* Gather data from bin to global worklist: thread-strided manner*/
	__forceinline__ __device__ void
	_thread_stride_gather(
		vertex_t *worklist,
		vertex_t *worklist_bin,
		vertex_t my_front_count,
		vertex_t output_off,
		const index_t bin_off)
	{
		// Thread strided global FQ generation
		// gather all frontiers into global mdata.worklist
		vertex_t dest_end = output_off + my_front_count;
		while (output_off < dest_end)
			worklist[output_off++] =
				worklist_bin[(--my_front_count) + bin_off];
	}
	__forceinline__ __device__ void
	_thread_stride_gather_element(
		vertex_t *worklist,
		vertex_t *worklist_bin,
		vertex_t my_front_count,
		vertex_t output_off,
		const index_t bin_off)
	{
		// Thread strided global FQ generation
		// gather all frontiers into global mdata.worklist
		vertex_t dest_end = output_off + my_front_count;
		while (output_off < dest_end)
			worklist[output_off++] =
				worklist_bin[(--my_front_count) + bin_off];
	}

	/* Gather data from bin to global worklist: warp-strided manner*/
	// template<typename vertex_t, typename index_t>
	__forceinline__ __device__ void
	_warp_stride_gather(
		vertex_t *worklist,
		vertex_t *worklist_bin,
		vertex_t my_front_count,
		vertex_t output_off,
		const index_t bin_off,
		const index_t WOFF)
	{
		vertex_t warp_front_count;
		index_t warp_input_off, warp_output_off, warp_dest_end;

		// Warp Stride
		for (int i = 0; i < 32; i++)
		{
			// Comm has problem.
			// if(__all(my_front_count * (i==WOFF)) == 0) continue;
			//
			// Quickly decide whether need to proceed on this thread
			warp_front_count = my_front_count;
			warp_front_count = __shfl_sync(0xffffffff, warp_front_count, i);
			if (warp_front_count == 0)
				continue;

			warp_output_off = output_off;
			warp_input_off = bin_off;
			warp_output_off = __shfl_sync(0xffffffff, warp_output_off, i);
			warp_input_off = __shfl_sync(0xffffffff, warp_input_off, i);
			warp_dest_end = warp_output_off + warp_front_count;
			warp_input_off += WOFF;
			warp_output_off += WOFF;

			while (warp_output_off < warp_dest_end)
			{
				worklist[warp_output_off] =
					worklist_bin[warp_input_off];
				warp_output_off += 32;
				warp_input_off += 32;
			}
		}
	}
	__forceinline__ __device__ void
	_warp_stride_gather_element(
		vertex_t *worklist,
		vertex_t *worklist_bin,
		vertex_t my_front_count,
		vertex_t output_off,
		const index_t bin_off,
		const index_t WOFF)
	{
		vertex_t warp_front_count;
		index_t warp_input_off, warp_output_off, warp_dest_end;

		// Warp Stride
		for (int i = 0; i < 32; i++)
		{

			warp_front_count = my_front_count;
			warp_front_count = __shfl_sync(0xffffffff, warp_front_count, i);
			if (warp_front_count == 0)
				continue;

			warp_output_off = output_off;
			warp_input_off = bin_off;
			warp_output_off = __shfl_sync(0xffffffff, warp_output_off, i);
			warp_input_off = __shfl_sync(0xffffffff, warp_input_off, i);
			warp_dest_end = warp_output_off + warp_front_count;
			warp_input_off += WOFF;
			warp_output_off += WOFF;

			while (warp_output_off < warp_dest_end)
			{
				worklist[warp_output_off] =
					worklist_bin[warp_input_off];
				warp_output_off += 32;
				warp_input_off += 32;
			}
		}
	}



	__forceinline__ __device__ void
	_push_coalesced_scan_random_list(
		const index_t TID,
		const index_t WIDL,
		const index_t WOFF,
		const index_t WCOUNT,
		const index_t GRNTY,
		feature_t level)
	{
		vertex_t my_front_sml = 0;
		vertex_t my_front_mid = 0;
		vertex_t my_front_lrg = 0;

		for (index_t my_beg = TID; my_beg < vert_count * VAL1; my_beg += GRNTY)
		{
			if (vert_status[my_beg]!=vert_status_prev[my_beg])
			{
				int v = my_beg / VAL1;
				index_t degree = beg_pos[v + 1] - beg_pos[v];
				if (degree == 0)
					continue;

				if (degree < SML_MID)
					my_front_sml++;
				else if (degree > MID_LRG)
					my_front_lrg++;
				else
					my_front_mid++;
			}
		}
		__syncthreads();
		vertex_t my_front_off_sml = 0;
		vertex_t my_front_off_mid = 0;
		vertex_t my_front_off_lrg = 0;

		// //For debugging
		cat_thd_count_sml[TID] = my_front_sml;
		cat_thd_count_mid[TID] = my_front_mid;
		cat_thd_count_lrg[TID] = my_front_lrg;

		// prefix-scan
		assert(WCOUNT >= 3);
		_grid_scan_agg<vertex_t, vertex_t>(WOFF, WIDL, WCOUNT,
										   my_front_sml,
										   my_front_mid,
										   my_front_lrg,
										   my_front_off_sml,
										   my_front_off_mid,
										   my_front_off_lrg,
										   worklist_sz_sml,
										   worklist_sz_mid,
										   worklist_sz_lrg);

		for (index_t my_beg = TID; my_beg < vert_count * VAL1; my_beg += GRNTY)
		{
			if (vert_status[my_beg]!=vert_status_prev[my_beg])
			{
				int v = my_beg / VAL1;
				index_t degree = beg_pos[v + 1] - beg_pos[v];
				if (degree == 0)
					continue;

				if (degree < SML_MID)
					worklist_sml[my_front_off_sml++] = my_beg;
				else if (degree > MID_LRG)
					worklist_lrg[my_front_off_lrg++] = my_beg;
				else
					worklist_mid[my_front_off_mid++] = my_beg;
			}

			if (my_beg < vert_count * VAL1)
				// make sure already activated ones are turned off
				if (vert_status_prev[my_beg] != vert_status[my_beg])
					vert_status_prev[my_beg] = vert_status[my_beg];
		}
		__syncthreads();
	}

	
};

#endif
