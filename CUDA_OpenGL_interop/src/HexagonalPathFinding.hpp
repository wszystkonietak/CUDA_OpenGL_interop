//#pragma once
//#include <iostream>
//#include <queue>
//#include <map>
//#include <unordered_map>
//#include <cmath>
//#include <string>
//#define INF 9999
//#define WIDTH 40
//#define HEIGHT 30
//
//using namespace std;
//
//class Node {
//public:
//    Node() {
//        x = 0;
//        y = 0;
//        h = 0;
//        g = 0;
//    };
//    Node(int pos_x, int pos_y) {
//        x = pos_x;
//        y = pos_y;
//    };
//    Node(int pos_x, int pos_y, float g_) {
//        x = pos_x;
//        y = pos_y;
//        g = g_;
//    };
//
//    Node* parent = NULL;
//    float h = 0;
//    int g = 0;
//    int x;
//    int y;
//
//    vector<Node> find_neighbors();
//    bool operator==(const Node& other) const
//    {
//        return (x == other.x && y == other.y);
//    };
//    string str() const {
//        return string("(") + std::to_string(this->y) + ", " + std::to_string(this->x) + ", g=" + std::to_string(this->g) + ", h=" + std::to_string(this->h) + ")";
//    };
//};
//
//struct KeyFuncs
//{
//    size_t operator()(const Node& k)const
//    {
//        return std::hash<int>()(10003 * k.x + 1 * k.y);
//    }
//
//    bool operator()(const Node& a, const Node& b)const
//    {
//        return a.x == b.x && a.y == b.y;
//    }
//};
//
//class NodeComparatorAstar {
//public:
//    //compare distance from start + distance to end from both nodes ant return information which one is closer
//    bool operator() (const Node& a, const Node& b) const
//    {
//        return  a.h + a.g > b.h + b.g;
//    }
//};
//
//class NodeComparatorDijkstra {
//public:
//    //compare two distances form start node and return information which one is closer
//    bool operator() (const Node& a, const Node& b) const
//    {
//        return  a.g > b.g;
//    }
//};
//
//class Dijkstra {
//public:
//    Dijkstra() {
//        start = Node(8, 4);
//        end = Node(WIDTH - 10, HEIGHT - 8);
//        node_size = 15;
//        break_beetween_nodes = 1;
//        all_node_size = node_size + break_beetween_nodes;
//        SCR_WIDTH = WIDTH * (node_size + break_beetween_nodes);
//        SCR_HEIGHT = HEIGHT * (node_size + break_beetween_nodes);
//
//        blockades_finished = false;
//    }
//    //variables used to calculate 
//    Node start;
//    Node end;
//    vector<Node> result;
//    unordered_map<Node, Node, KeyFuncs> paths;
//    unordered_map<Node, int, KeyFuncs> visited;
//    priority_queue<Node, vector<Node>, NodeComparatorAstar> open;
//    vector<Node> blockades;
//    bool blockades_finished;
//
//
//    //variables used for printing
//    int node_size;
//    int break_beetween_nodes;
//    int all_node_size;
//    int SCR_WIDTH;
//    int SCR_HEIGHT;
//
//    void find_path();
//    void count_result();
//    void find_parent(Node n);
//    void print_data();
//    std::vector<int> getEdgeNeighboursFromId(int id);
//    std::vector<int> getEdgesFromCellId(Cell cell);
//};
