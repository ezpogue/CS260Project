import random
import sys
import numpy as np
from utils import *


def parse_map(filename):
    try:
        with open(filename, 'r') as file:
            dimensions = file.readline().strip().split(' ')
            rows = int(dimensions[0])
            columns = int(dimensions[1])

            result = []

            for row in range(rows):
                line = file.readline().strip().split(' ')
                row_list = []
                for col in range(columns):
                    item = tuple(line[col].split(','))
                    row_list.append(item)
                result.append(row_list)

            return result

    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def find_houses(map):
    coords = []
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j][0] == 'H':
                coords.append((i,j))
    return coords

def distance(source, destination):
    return ((source[0] - destination[0])**2 + (source[1] - destination[1])**2)**.5

def valid(next, position, map):
    if not (0 <= next[0] < len(map)):
        return False
    if not (0 <= next[1] < len(map[0])):
        return False
    if map[next[0]][next[1]][0] == 'W':
        return False
    if map[next[0]][next[1]][1] != map[position[0]][position[1]][1]:
        return False
    return True

def cost(source, destination):
    return 1

def rand_nn_st(map):
    houses = find_houses(map)
    print(houses)
    visited = []
    visited.append(random.choice(houses))

    road = []
    for row in range(len(map)):
            road.append([0] * len(map[0]))
    while len(visited) != len(houses):
        source = random.choice(visited)
        mindist = len(map) * 2
        for house in houses:
            if house == source or house in visited:
                continue
            dist = distance(source, house)
            if dist < mindist:
                mindist = dist
                destination = house
        backtrack = []
        for row in range(len(map)):
            backtrack.append([(0,0)] * len(map[0]))
        closed = set()
        open = PriorityQueue(order=min, f=lambda v: v.f)
        open.put(source, Value(distance(source, destination), 0))
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        while len(open) > 0:
            node, val = open.pop()
            closed.add(node)
            if node == destination:
                break
            for move in moves:
                next = (move[0] + node[0], move[1] + node[1])
                if next not in closed and valid(next, node, map):
                    g = val.g + cost(node, next)
                    f = g + distance(next, destination) 
                    open.put(next, Value(f, g))
                    backtrack[next[0]][next[1]] = node
        
        path = [destination]
        curr = destination
        while curr != source:
            curr = backtrack[curr[0]][curr[1]]
            path = [curr] + path

        for tile in path:
            road[tile[0]][tile[1]] = 1
        print(road)

        visited.append(destination)

    return road
        
map = parse_map('largertestmap.txt')
road = rand_nn_st(map)
print(map)
print(np.matrix(road))

        
    
