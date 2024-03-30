#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "DataTypes.hpp"

struct ConstantsInitData {
	float diameter;
	float max_distance_squared;
	float radius;
	unsigned int x_shift;
	unsigned int objects_size;
};

struct ConstantsRadixSetUp {
	unsigned int groups_per_block;
	unsigned int threads_per_group;
	unsigned int shared_memory_size;
	unsigned int cells_per_group;
	unsigned int num_radices;
	unsigned int cells_size;
};

struct ConstantsRadixSum {
	unsigned int radices_per_block;
	unsigned int num_radices;
	unsigned int num_groups;
	unsigned int padded_groups;
};

struct ConstantsRadixReorder {
	unsigned int groups_per_block;
	unsigned int threads_per_group;
	unsigned int num_radices;
	unsigned int cells_per_group;
	unsigned int radices_per_block;
	unsigned int padded_blocks;
	unsigned int cells_size;
};

struct ConstantsCellColide {
	float diameter_squared;
	float radius;
	unsigned int objects_size;
	float scene_width;
	float scene_hieght;
};

class CollisionDetection {
public:
	CollisionDetection(float width, float height, unsigned int size, float radius, Particle* d_particles);
	CollisionDetection() {};
	//methods
	void setup(float scene_width, float scene_height, unsigned int size, float radius, Particle* d_particles);
	void check_collision();
	//members
	unsigned int objects_size;
	unsigned int cells_size;
	float scene_width;
	float scene_height;
	unsigned int width;
	unsigned int height;
	unsigned int num_blocks;
	unsigned int num_threads;
	unsigned int bit_step_size;
	unsigned int groups_per_block;
	unsigned int threads_per_group;
	unsigned int num_radices;
	unsigned int num_radicesnum_threads;
	unsigned int num_groups;
	unsigned int padded_blocks;
	unsigned int padded_groups;
	unsigned int part_size;
	unsigned int radices_per_block;
	unsigned int radices_size;
	unsigned int cells_per_group;
	unsigned int shared_memory_size;
	unsigned int cells_per_thread;
	unsigned int h_cell_count;
	unsigned int x_shift;
	unsigned int min_bits_for_hash;
	unsigned int shared_size;
	unsigned int collisions_count;
	unsigned int zero;
	float radius;
	float diameter;
	float diameter_squared;
	float max_distance_squared;
	//kernel constants
	ConstantsCellColide constantsCellColide;
	ConstantsRadixSetUp constantsRadixSetUp;
	ConstantsRadixSum constantsRadixSum;
	ConstantsRadixReorder constantsRadixReorder;
	ConstantsInitData constantsInitData;
	//device members
	unsigned int* d_radices;
	unsigned int* d_radices_prefix_sum;
	unsigned int* d_cells;
	unsigned int* d_cells_tmp;
	unsigned int* cells_swap;
	unsigned int* d_objects;
	unsigned int* d_objects_tmp;
	unsigned int* objects_swap;
	unsigned int* d_cell_count;
	unsigned int* d_collisions_count;
	unsigned int* d_particles_coliding_indices;
	Particle* d_particles;
private:
	unsigned int succesive_power_of_two(unsigned int n);
	unsigned int count_bits(unsigned int n);
};


