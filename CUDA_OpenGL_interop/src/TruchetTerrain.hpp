#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Shader.hpp"
#include "ComputeShader.hpp"
#include "Surface.hpp"
#include "RandomSeed.hpp"
#include <algorithm>
#include <iostream>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <math.h>
#include <glm/glm.hpp>

struct HexagonData {
	HexagonData(int id) : id(id) {
		for (int i = 0; i < 3; i++) {
			flip_x[i] = 0;
			flip_y[i] = 0;
			offset_x[i] = 0;
			multiply_x[i] = 0;
		}
	}
	int id;
	float flip_y[3];
	float flip_x[3];
	float offset_x[3];
	float multiply_x[3];
};

struct GenerateTerrainUniform {
	glm::vec2 resolution;
	glm::vec2 board_size;
};

struct Cell {
	Cell() : x(0), y(0) {}
	Cell(int x, int y) : x(x), y(y) {}
	int x;
	int y;
	bool operator==(const Cell& other) const {
		return x == other.x && y == other.y;
	}
	bool operator==(const int& other) const {
		return x == other && y == other;
	}
	bool operator<(const Cell& other) const {
		return (x < other.x) || (x == other.x && y < other.y);
	}
};

struct Hexagon {
	int edges[6];
	int entrances;
};

class HexagonEdges {
public:
	HexagonEdges() {
		edges[0] = Cell(INT_MAX, INT_MAX);
		edges[1] = Cell(INT_MAX, INT_MAX);
		edges[2] = Cell(INT_MAX, INT_MAX);
		id = -1;
	}
	HexagonEdges(Cell e1, Cell e2, Cell e3) {
		edges[0] = e1;
		edges[1] = e2;
		edges[2] = e3;
		generate_id();
	}
	void generate_id() {
		if (edges[1].x > edges[2].x) {
			Cell tmp = edges[1];
			edges[1] = edges[2];
			edges[2] = tmp;
		}
		if (edges[0].x > edges[1].x) {
			Cell tmp = edges[0];
			edges[0] = edges[1];
			edges[1] = tmp;
		}
		if (edges[1].x > edges[2].x) {
			Cell tmp = edges[1];
			edges[1] = edges[2];
			edges[2] = tmp;
		}
		id = edges[0].x;
		id |= (edges[0].y << 3);
		id |= (edges[1].x << 6);
		id |= (edges[1].y << 9);
		id |= (edges[2].x << 12);
		id |= (edges[2].y << 15);
	}
	Cell edges[3];
	int id;
	bool operator==(const HexagonEdges& other) const {
		return edges[0] == other.edges[0] && edges[1] == other.edges[1] && edges[2] == other.edges[2];
	}
};

struct HexEdgeHash
{
	size_t operator()(const HexagonEdges& h)const
	{
		return std::hash<int>()(h.id);
	}
};

struct InCellPos {
	float x, y;
};



class TruchetTerrain {
public:
	TruchetTerrain() = default;
	TruchetTerrain(std::string&& shaders_path) { setup(std::move(shaders_path)); }
	void setup(std::string&& shaders_path);
	InCellPos getClosestPoint(Cell cell, int entry_side, InCellPos pos);
	void update();
	void draw();
	void reloadShader(std::string&& path);
private:
	void generateBoard();
	void generateCellsIndices();
	void generateDetails();
	
	std::vector<Cell> getCellsFromEdgeId(int id);
	std::vector<int> getEdgeNeighboursFromEdgeId(int id);
	std::vector<int> getEdgeNeighboursFromEdgeId(int id, Cell& current_cell);
	std::vector<int> getEdgesFromCellId(Cell id);
	bool isValidEdge(int edge, Cell current_cell, int current_edge);
	bool isValidEdge(int edge, Cell current_cell, std::set<Cell>& prev_cells, int current_edge = INT_MAX, bool is_first = false);
	void addPathToBoard(int edge1, int edge2, Cell cell);
	void generateHexIds();
	void generateLookupEdges();
	void printBoard();
	//tests long path when result is false for size (4, 4) should be false because not all cells are filled
	void setBoardTestCase1();
	//tests if path covers 100% of the cell for size(4, 4) should give false for (61, 3, 3) and true for (62 and 54, 3, 3) if there is no option of returning to this cell and you are leaving
	//tests long path when result if true for size(4, 4) should be true because all other cells are filled and this is last path to end 
	void setBoardTestCase2();
	//tests if path covers 100% of the board for size(4, 5) path should give false because it cuts connection for part of the board
	void setBoardTestCase3();
	
	Cell size;
	Cell beggining;
	Hexagon* board;
	HexagonEdges* hex_edges;
	std::map<int, int> entrances;
	std::map<int, int> path;
	std::unordered_map<HexagonEdges, int, HexEdgeHash> lookup_edges;
	std::vector<Cell> cells_path;
	int edges_size;
	int edge_row;
	int wall;
	int railing;
	int finish_edge;
	std::vector<int> cellEdgesOnGridEdges;
	//for drawing
	Surface<float4> d_canvas;
	GLuint canvas;
	GLuint hex_ids_ssbo;
	GenerateTerrainUniform generate_canvas_uniform;
	GLuint generate_canvas_ubo;
	Shader canvas_shader;
	ComputeShader generate_canvas_shader;
	std::vector<HexagonData> hex_ids;
	std::vector<int> edges_data;
	std::string str;
	uint2 resolution;
	dim3 block_size;
	dim3 grid_size;
	unsigned int quadvao = 0, quadvbo = 0;
};