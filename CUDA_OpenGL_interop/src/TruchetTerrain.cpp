#include "TruchetTerrain.hpp"

void printVector(const std::vector<int>& vec) {
	for (int num : vec) {
		std::cout << num << " ";
	}
	std::cout << std::endl;
}

void printVector(const std::vector<Cell>& vec) {
	for (const Cell& cell: vec) {
		std::cout << "(" << cell.x << ", " << cell.y << ") ";
	}
	std::cout << std::endl;
}

void TruchetTerrain::setup() 
{
	size.x = 4;
	size.y = 4;

	board = new Hexagon[size.x * size.y];

	edges_size = (size.y - 1) * ((size.x * 2 + 1) + (size.x + 1)) + (size.x + 1) + 2 * (size.x * 2);
	edge_row = size.x * 2;
	wall = (size.x + 1);
	railing = (edge_row + 1);


	generateCellsIndices();
	generateDetails();
}

void TruchetTerrain::generateCellsIndices() 
{
	generateBoard();
	Cell current_cell(0, 0);
	int current_edge = 0;
	int current_edge_id = 0;
	std::vector<int> edges;
	std::vector<int> edges_ids;
	int number = 0;
	while (current_edge != finish_edge) {
		std::cout << "\n\n****************************************\n\n";

		std::cout << "current_cell: (" << current_cell.x << ", " << current_cell.y << ")\n";
		std::cout << "current_edge: " << current_edge << "  current_edge_id: " << current_edge_id << '\n';
		printBoard();

		edges = getEdgeNeighboursFromEdgeId(current_edge, current_cell);
		edges_ids.clear();
		std::cout << "before validation\n";
		printVector(edges);
		for (int i = 0; i < 6; i++) {
			if (current_edge_id != i && !(board[current_cell.y * size.x + current_cell.x].edges[i] % 2)) {
				if (isValidEdge(edges[i], current_cell, current_edge)) {
					edges_ids.push_back(i);
				}
			}
		}
		std::cout << "\nafter validation\n";
		for (int& i : edges_ids) {
			std::cout << edges[i] << " ";
		}
		std::cout << '\n';
		/*std::cout << "pick edge index: \n";
		std::cin >> number;
		if (number > edges_ids.size()) {
			std::cin >> number;
		}*/
		if (edges_ids.size() > 1) {
			std::uniform_real_distribution<> rand_id(0, edges_ids.size() - 1);
			number = rand_id(gen);
		}
		else {
			//std::uniform_real_distribution<> rand_id(0, edges_ids.size());
			number = 0;
		}
		
		if (edges_ids.size() == 0) {
			generateBoard();
		}
		addPathToBoard(current_edge_id, edges_ids[number], current_cell);
		current_edge = edges[edges_ids[number]];

		if (current_edge == finish_edge) { break; }

		if (getCellsFromEdgeId(current_edge).size() > 1) {
			current_edge_id = 5 - edges_ids[number];
		}
		else {
			current_edge_id = edges_ids[number];
		}
		if (entrances.contains(current_edge)) {
			if (current_cell != getCellsFromEdgeId(entrances[current_edge])[0]) {
				current_cell = getCellsFromEdgeId(entrances[current_edge])[0];
			}
			current_edge = entrances[current_edge];
			for (int i = 0; i < 6; i++) {
				if (current_edge == (board[current_cell.y * size.x + current_cell.x].edges[i] >> 1)) {
					current_edge_id = i;
				}
			}
		}
	}
}

void TruchetTerrain::generateDetails() 
{

}

void TruchetTerrain::generateBoard()
{
	std::vector<int> tmp;
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			board[y * size.x + x].entrances = 6;
			tmp = getEdgesFromCellId(Cell(x, y));
			for (int i = 0; i < 6; i++) {
				board[y * size.x + x].edges[i] = tmp[i] << 1 | 0x00;
			}
		}
	}
	int edges_size_2 = 0;
	int board_edges_size = size.x * 4 + size.y * 4 - 2;
	int* board_edges_indices = new int[board_edges_size];
	const int top_offset = 0;
	const int right_offset = size.x * 2;
	const int bottom_offset = size.x * 2 + size.y * 2 - 1;
	const int left_offset = size.x * 4 + size.y * 2 - 1;

	int top_count = 0,
		bottom_count = size.x * 2 - 1,
		left_count = size.y * 2 - 2,
		right_count = 0;

	for (int i = 0; i < edges_size; i++) {
		//top row
		if (i < edge_row) {
			board_edges_indices[top_offset + top_count] = i;
			top_count++;
			Cell c = getCellsFromEdgeId(i)[0];
			board[c.y * size.x + c.x].entrances--;
			cellEdgesOnGridEdges.push_back(i);
			continue;
		}
		//bottom row
		if (i >= edges_size - edge_row) {
			board_edges_indices[bottom_offset + bottom_count] = i;
			bottom_count--;
			Cell c = getCellsFromEdgeId(i)[0];
			board[c.y * size.x + c.x].entrances--;
			cellEdgesOnGridEdges.push_back(i);
			continue;
		}
		//left side
		if (!((i - edge_row) % (wall + railing))) {
			board_edges_indices[left_offset + left_count] = i;
			left_count--;
			Cell c = getCellsFromEdgeId(i)[0];
			board[c.y * size.x + c.x].entrances--;
			cellEdgesOnGridEdges.push_back(i);
			continue;
		}
		//right side
		if (((i - edge_row) % (wall + railing)) == wall - 1) {
			board_edges_indices[right_offset + right_count] = i;
			right_count++;
			Cell c = getCellsFromEdgeId(i)[0];
			board[c.y * size.x + c.x].entrances--;
			cellEdgesOnGridEdges.push_back(i);
			continue;
		}
		//inclined
		if (((i - edge_row) % (wall + railing)) == wall) {
			board_edges_indices[left_offset + left_count] = i;
			left_count--;
			Cell c = getCellsFromEdgeId(i)[0];
			board[c.y * size.x + c.x].entrances--;
			cellEdgesOnGridEdges.push_back(i);
			continue;
		}
		if (((i - edge_row) % (wall + railing)) == wall + railing - 1) {
			board_edges_indices[right_offset + right_count] = i;
			right_count++;
			Cell c = getCellsFromEdgeId(i)[0];
			board[c.y * size.x + c.x].entrances--;
			cellEdgesOnGridEdges.push_back(i);
			continue;
		}
	}
	//only for beggining
	board[0].entrances++;
	for (int i = 0; i < board_edges_size; i++) {
		if (i % 2) {
			entrances[board_edges_indices[i]] = board_edges_indices[i - 1];
		}
		else {
			entrances[board_edges_indices[i]] = board_edges_indices[(i + 1) % board_edges_size];
		}
	}
	beggining = Cell(0, 0);
	finish_edge = entrances[0];
	for (auto& [key, value] : entrances) {
		Cell c1 = getCellsFromEdgeId(key)[0];
		Cell c2 = getCellsFromEdgeId(value)[0];
		if (!(c1.x == c2.x && c1.y == c2.y)) {
			board[c1.y * size.x + c1.x].entrances++;
		}
	}
	//generateCellsIndices();
	//setBoardTestCase3();
}

std::vector<Cell> TruchetTerrain::getCellsFromEdgeId(int id)
{
	std::vector<Cell> result;
	int column_offset = ((id - edge_row) % (wall + railing));
	int row_offset = ((id - edge_row) / (wall + railing));

	bool isWall = column_offset < wall && column_offset > -1;
	//bool isEdgeRow = edges_size - id - 1 < edge_row || id < edge_row;
	Cell cell;
	cell.y = row_offset;
	cell.x = isWall ? column_offset : id < edge_row ? id / 2 : (column_offset - wall) / 2;
	/*if (!(cell.x < 0 || cell.x > size.x - 1 || cell.y < 0 || cell.y > size.y - 1)) {
		result.push_back(cell);
	}*/
	if (isWall) {	
		if (cell.x < size.x) {
			result.push_back(cell);
		}
		if (cell.x - 1 > -1) {
			result.push_back(Cell(cell.x - 1, cell.y));
		}
	}
	else if (id < edge_row) {
		result.push_back(cell);
	}
	else if (id > edges_size - edge_row - 1) {
		result.push_back(cell);
	}
	else {
		bool isRowEven = !(size.x % 2);
		bool isLeft = true;
		bool columnNotEvenLeft = false;
		bool columnNotEvenRight = false;
		if (isRowEven) {
			if (cell.y % 2 && !((column_offset - wall) % 2)) {
				columnNotEvenLeft = true;
			}
			isLeft = (column_offset - wall) % 2;
		}
		else {
			if (cell.y % 2) {
				if (((column_offset - wall) % 2)) {
					columnNotEvenRight = true;
				}
				else {
					columnNotEvenLeft = true;
				}
			}
			isLeft = ((column_offset + (cell.y % 2)) - wall) % 2;
		}
		if (isLeft) {
			if (columnNotEvenLeft) {
				if (cell.x < size.x) {
					result.push_back(Cell(cell.x, cell.y + 1));
				}
				if (cell.x - 1 > -1) {
					result.push_back(Cell(cell.x - 1, cell.y));
				}
			}
			else {
				if (cell.y + 1 < size.y) {
					result.push_back(cell);
					result.push_back(Cell(cell.x, cell.y + 1));
				}
			}
		}
		else {
			if (columnNotEvenLeft) {
				if (cell.x < size.x) {
					result.push_back(Cell(cell.x, cell.y + 1));
				}
				if (cell.x - 1 > -1) {
					result.push_back(Cell(cell.x - 1, cell.y));
				}
			}
			else {
				if (columnNotEvenRight) {
					result.push_back(cell);
					if (cell.y + 1 < size.y) {
						result.push_back(Cell(cell.x, cell.y + 1));
					}
				}
				else {
					if (cell.x < size.x) {
						result.push_back(cell);
					}
					if (cell.y + 1 < size.y && cell.x - 1 > -1) {
						result.push_back(Cell(cell.x - 1, cell.y + 1));
					}
				}
			}
			
		}
	}
	//printVector(result);
	return result;
}

std::vector<int> TruchetTerrain::getEdgeNeighboursFromEdgeId(int id)
{
	std::vector<Cell> cells = getCellsFromEdgeId(id);
	std::vector<int> neighbours;
	std::vector<int> tmp;
	for (Cell& cell : cells) {
		tmp = getEdgesFromCellId(cell);
		neighbours.insert(neighbours.end(), tmp.begin(), tmp.end());
	}
	neighbours.erase(std::remove(neighbours.begin(), neighbours.end(), id), neighbours.end());
	printVector(neighbours);
	return neighbours;
}

std::vector<int> TruchetTerrain::getEdgeNeighboursFromEdgeId(int id, Cell& current_cell)
{
	std::vector<Cell> cells = getCellsFromEdgeId(id);
	std::vector<int> neighbours;
	std::vector<int> tmp;
	Cell updated_cell = current_cell;
	if (cells.size() > 1) {
		for (Cell& cell : cells) {
			if (!(cell.x == current_cell.x && cell.y == current_cell.y)) {
				updated_cell = cell;
				tmp = getEdgesFromCellId(cell);
				neighbours.insert(neighbours.end(), tmp.begin(), tmp.end());
			}
		}
	} else {
		tmp = getEdgesFromCellId(cells[0]);
		neighbours.insert(neighbours.end(), tmp.begin(), tmp.end());
	}
	current_cell = updated_cell;
	//neighbours.erase(std::remove(neighbours.begin(), neighbours.end(), id), neighbours.end());
	//printVector(neighbours);
	return neighbours;
}

std::vector<int> TruchetTerrain::getEdgesFromCellId(Cell cell)
{
	if (cell.x < 0 || cell.x > size.x - 1 || cell.y < 0 || cell.y > size.y - 1) {
		return std::vector<int>();
	}
	std::vector<int> neighbours;
	if (!cell.y) {
		neighbours.push_back(cell.x * 2);
		neighbours.push_back(cell.x * 2 + 1);
		neighbours.push_back(edge_row + cell.x);
		neighbours.push_back(edge_row + cell.x + 1);
		neighbours.push_back(edge_row + wall + cell.x * 2);
		neighbours.push_back(edge_row + wall + cell.x * 2 + 1);
		return neighbours;
	}
	if (cell.y % 2) {
		neighbours.push_back(edge_row + cell.y * (wall + railing) - railing + 1 + cell.x * 2);
		neighbours.push_back(edge_row + cell.y * (wall + railing) - railing + 1 + cell.x * 2 + 1);
		neighbours.push_back(edge_row + cell.y * (wall + railing) + cell.x);
		neighbours.push_back(edge_row + cell.y * (wall + railing) + cell.x + 1);
		if (cell.y == size.y - 1) {
			neighbours.push_back(edge_row + cell.y * (wall + railing) + wall + cell.x * 2);
			neighbours.push_back(edge_row + cell.y * (wall + railing) + wall + cell.x * 2 + 1);
		}
		else {
			neighbours.push_back(edge_row + cell.y * (wall + railing) + wall + 1 + cell.x * 2);
			neighbours.push_back(edge_row + cell.y * (wall + railing) + wall + 1 + cell.x * 2 + 1);
		}
		return neighbours;
	}
	else {
		neighbours.push_back(edge_row + cell.y * (wall + railing) - railing + cell.x * 2);
		neighbours.push_back(edge_row + cell.y * (wall + railing) - railing + cell.x * 2 + 1);
		
		neighbours.push_back(edge_row + cell.y * (wall + railing) + cell.x);
		neighbours.push_back(edge_row + cell.y * (wall + railing) + cell.x + 1);

		neighbours.push_back(edge_row + cell.y * (wall + railing) + wall + cell.x * 2);
		neighbours.push_back(edge_row + cell.y * (wall + railing) + wall + cell.x * 2 + 1);
		return neighbours;
	}
	return neighbours;
}

bool TruchetTerrain::isValidEdge(int edge, Cell current_cell, int current_edge)
{
	if (edge == current_edge) {
		return false;
	}
	std::set<Cell> set;
	set.insert(current_cell);
	return isValidEdge(edge, current_cell, set, current_edge, true);
}
//TEST TEST TEST TEST
bool TruchetTerrain::isValidEdge(int edge, Cell current_cell, std::set<Cell>& prev_cells, int current_edge, bool is_first)
{
	std::cout << 0 << ' ';
	//when you are in last cell and you can go somewhere else
	if (edge == finish_edge && board[current_cell.y * size.x + current_cell.x].entrances > 1) {
		return false;
	}
	//when you dont have option to return to cell but this edge is leaving
	if (is_first && (current_cell.x == 0 || current_cell.x == size.x - 1 || current_cell.y == 0 || current_cell.y == size.y - 1) && current_cell != Cell(0, 0)) {
		if (board[current_cell.y * size.x + current_cell.x].entrances < 3) {
			int options_count = 0;
			for (int i = 0; i < 6; i++) {
				if (!(board[current_cell.y * size.x + current_cell.x].edges[i] % 2)) {
					options_count++;
				}
			}
			if (options_count > 2) {
				if (entrances.contains(edge)) {
					if (getCellsFromEdgeId(entrances[edge])[0] == current_cell) {
						return true;
					}
				}
				return false;
			}
		}
	}
	std::cout << 1 << ' ';
	//find second cell with this edge
	bool is_border_edge = false;
	std::vector<Cell> tmp = getCellsFromEdgeId(edge);
	Cell c;
	if (tmp.size() == 1) {
		is_border_edge = true;
		edge = entrances[edge];
		c = getCellsFromEdgeId(edge)[0];
	}
	else {
		for (auto& cell : tmp) {
			if (!(cell.x == current_cell.x && cell.y == current_cell.y)) {
				c = cell;
			}
		}
	}
	//when there is multiple options in your cell
	if (board[c.y * size.x + c.x].entrances > 2) {
		return true;
	}
	std::cout << 2 << ' ';
	//when you are in last cell and its only one option 
	if (c.x == beggining.x && c.y == beggining.y) {
		//not nesesary when cell is (0, 0)
		if (board[c.y * size.x + c.x].entrances > 1) {
			return true;
		}
		std::cout << 3 << ' ';
		prev_cells.insert(c);
		for (int y = 0; y < size.y; y++) {
			for (int x = 0; x < size.x; x++) {
				if (board[y * size.x + x].entrances > 0 && !prev_cells.contains(Cell(x, y))) {
					return false;
				}
			}
		}
		std::cout << 4 << ' ';
		return true;
	}
	//when you are not in last cell and there is only one option
	for (int i = 0; i < 6; i++) {
		if ((board[c.y * size.x + c.x].edges[i] >> 1) != edge) {
			if (!(board[c.y * size.x + c.x].edges[i] % 2)) {
				if (entrances.contains(board[c.y * size.x + c.x].edges[i] >> 1)) {
					if (c != getCellsFromEdgeId(entrances[board[c.y * size.x + c.x].edges[i] >> 1])[0]) {
						prev_cells.insert(c);
						std::cout << 5 << " " << (board[c.y * size.x + c.x].edges[i] >> 1) << "\n";
						return isValidEdge(board[c.y * size.x + c.x].edges[i] >> 1, c, prev_cells);
					}
				}
				else {
					prev_cells.insert(c);
					std::cout << 5 << " " << (board[c.y * size.x + c.x].edges[i] >> 1) << "\n";
					return isValidEdge(board[c.y * size.x + c.x].edges[i] >> 1, c, prev_cells);
				}
			}
		}
		/*if ((board[c.y * size.x + c.x].edges[i] >> 1) != edge) {
			if (!(board[c.y * size.x + c.x].edges[i] % 2)) {
				if (is_border_edge) {
					if (getCellsFromEdgeId(entrances[(board[c.y * size.x + c.x].edges[i] >> 1)])[0] != c && board[c.y * size.x + c.x].edges[i] >> 1 != current_edge) {
						prev_cells.insert(c);
						return isValidEdge(board[c.y * size.x + c.x].edges[i] >> 1, c, prev_cells);
					}
					else {
						continue;
					}
				}
				prev_cells.insert(c);
				return isValidEdge(board[c.y * size.x + c.x].edges[i] >> 1, c, prev_cells);
			}
		}*/
	}
}

void TruchetTerrain::addPathToBoard(int edge1, int edge2, Cell cell)
{
	path[board[cell.y * size.x + cell.x].edges[edge1] >> 1] = board[cell.y * size.x + cell.x].edges[edge2] >> 1;
	board[cell.y * size.x + cell.x].edges[edge1] |= 0x01;
	board[cell.y * size.x + cell.x].edges[edge2] |= 0x01;
	if (board[cell.y * size.x + cell.x].edges[edge1] >> 1 == 0) {
		board[cell.y * size.x + cell.x].entrances--;
	}
	if (entrances.contains(board[cell.y * size.x + cell.x].edges[edge1] >> 1)) {
		if (getCellsFromEdgeId(entrances[board[cell.y * size.x + cell.x].edges[edge1] >> 1])[0] != cell) {
			board[cell.y * size.x + cell.x].entrances--;
		}
	}
	else {
		board[cell.y * size.x + cell.x].entrances--;
	}
	if (entrances.contains(board[cell.y * size.x + cell.x].edges[edge2] >> 1)) {
		if (getCellsFromEdgeId(entrances[board[cell.y * size.x + cell.x].edges[edge2] >> 1])[0] != cell) {
			board[cell.y * size.x + cell.x].entrances--;
		}
	}
	else {
		board[cell.y * size.x + cell.x].entrances--;
	}
}

void TruchetTerrain::printBoard() {
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			std::cout << "(" << x << ", " << y << ") -> " << board[y * size.x + x].entrances<<" edges: ";
			for (int i = 0; i < 6; i++) {
				if (!(board[y * size.x + x].edges[i] % 2)) {
					std::cout << (board[y * size.x + x].edges[i] >> 1) << " ";
				}
			}
			std::cout << '\n';
		}
	}
}

void TruchetTerrain::setBoardTestCase1()
{
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			if (x == 0 && y == 0) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 1;
			}
			if (x == 0 && y == 1) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			if (x == 0 && y == 2) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			if (x == 3 && y == 2) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			if (x == 0 && y == 3) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			if (x == 1 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			if (x == 2 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			if (x == 3 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
		}
	}
	printBoard();
}

void TruchetTerrain::setBoardTestCase2()
{
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			if (x == 0 && y == 0) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 1;
			}
			else if (x == 0 && y == 1) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 0 && y == 2) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			else if (x == 3 && y == 2) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 0 && y == 3) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			else if (x == 1 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 2 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 3 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			else {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].entrances = 0;
			}
		}
	}
	printBoard();
}

void TruchetTerrain::setBoardTestCase3()
{
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			if (x == 0 && y == 0) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 1;
			}
			else if (x == 0 && y == 1) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 0 && y == 2) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			else if (x == 3 && y == 2) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 0 && y == 3) {
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			else if (x == 1 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 2 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].entrances -= 4;
			}
			else if (x == 3 && y == 3) {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].entrances -= 2;
			}
			else if (y == 4) {
				continue;
			}
			else {
				board[y * size.x + x].edges[0] |= 0x01;
				board[y * size.x + x].edges[1] |= 0x01;
				board[y * size.x + x].edges[2] |= 0x01;
				board[y * size.x + x].edges[3] |= 0x01;
				board[y * size.x + x].edges[4] |= 0x01;
				board[y * size.x + x].edges[5] |= 0x01;
				board[y * size.x + x].entrances = 0;
			}
		}
	}
	printBoard();
}

