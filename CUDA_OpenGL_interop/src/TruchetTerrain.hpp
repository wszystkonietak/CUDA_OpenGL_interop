#pragma once

#include "RandomSeed.hpp"
#include <algorithm>
#include <iostream>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <cmath>

#define EDGE_NOT_USED INT_MAX

struct Cell {
	Cell() : x(0), y(0) {}
	Cell(int x, int y) : x(x), y(y) {}
	int x;
	int y;
	bool operator==(const Cell& other) const {
		return x == other.x && y == other.y;
	}
	bool operator<(const Cell& other) const {
		return (x < other.x) || (x == other.x && y < other.y);
	}
};

struct Hexagon {
	int edges[6];
	int entrances;
};

struct InCellPos {
	float x, y;
};

class TruchetTerrain {
public:
	TruchetTerrain() { setup(); }
	void setup();
	InCellPos getClosestPoint(Cell cell, int entry_side, InCellPos pos);
	void generateCellsIndices();
	void generateDetails();
private:
	void generateBoard();
	
	std::vector<Cell> getCellsFromEdgeId(int id);
	std::vector<int> getEdgeNeighboursFromEdgeId(int id);
	std::vector<int> getEdgeNeighboursFromEdgeId(int id, Cell& current_cell);
	std::vector<int> getEdgesFromCellId(Cell id);
	bool isValidEdge(int edge, Cell current_cell, int current_edge);
	bool isValidEdge(int edge, Cell current_cell, std::set<Cell>& prev_cells, int current_edge = INT_MAX, bool is_first = false);
	void addPathToBoard(int edge1, int edge2, Cell cell);
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
	std::map<int, int> entrances;
	std::map<int, int> path;
	int edges_size;
	int edge_row;
	int wall;
	int railing;
	int finish_edge;
	std::vector<int> cellEdgesOnGridEdges;
};