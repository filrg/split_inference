from collections import deque


class Dijkstra :
    def __init__(self , cost , machine):
        self.cost = cost
        self.num_points = len(cost)//2 - 1
        self.machine = machine

        # config node
        self.start = 1

    def dijkstra(self ):
        visited = set()
        node = {"node": self.start, "prev": -1, "cost": 0}
        heap = deque([node])
        store = deque()

        while len(heap) > 0:
            # pick min-cost node
            # print("heap : " ,  heap)
            min_cost = min(item["cost"] for item in heap)
            for item in list(heap):
                if item["cost"] == min_cost:
                    current_node = item
                    heap.remove(item)
                    break

            # mark visited
            visited.add(current_node["node"])
            store.append(current_node)

            # relax neighbors
            for j, edge_cost in enumerate(self.cost[current_node["node"]]):
                if edge_cost != -1 and j not in visited:
                    new_node = {
                        "node": j,
                        "prev": current_node["node"],
                        "cost": current_node["cost"] + edge_cost
                    }

                    exist = False
                    for item in heap:
                        if item["node"] == new_node["node"]:
                            exist = True
                            if item["cost"] > new_node["cost"]:
                                item["cost"] = new_node["cost"]
                                item["prev"] = new_node["prev"]
                    if not exist:
                        heap.append(new_node)

        return list(store)

    def run(self):
        trace_back = self.dijkstra()
        end_node = self.num_points * 2
        curr_node = end_node
        lst_nodes = []
        while True:
            lst_nodes.append(curr_node)
            if curr_node == self.start:
                break
            for item in trace_back:
                if item["node"] == curr_node:
                    curr_node = item["prev"]
                    break

        lst_nodes.reverse()
        cut_point = -1
        for i in range(len(lst_nodes)):
            if lst_nodes[i] > self.num_points:
                cut_point = i
                break

        # print(self.machine[0] , " : ", lst_nodes[:cut_point])
        # lst_nodes = [x - self.num_points for x in lst_nodes]
        # print(self.machine[1] , " : ",lst_nodes[cut_point :])
        return cut_point + 1


