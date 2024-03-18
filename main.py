import random
import sys
import numpy as np
from utils import *
import time


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
    return (abs(source[0] - destination[0]) + abs(source[1] - destination[1]))

def valid(next, position, map):
    if not (0 <= next[0] < len(map)):
        return False
    if not (0 <= next[1] < len(map[0])):
        return False
    if map[next[0]][next[1]][0] == 'W':
        return False
    if next != position and abs(int(map[next[0]][next[1]][1]) - int(map[position[0]][position[1]][1])) > 1:
        return False
    return True

def cost(source, destination, map, road):
    if road[destination[0]][destination[1]] == 1:
        return 0
    if map[source[0]][source[1]][1] != map[destination[0]][destination[1]][1]:
        return 2
    return 1

def ARINS_st(map):
    houses = find_houses(map)
    visited = []
    visited.append(random.choice(houses))

    road = []
    for row in range(len(map)):
            road.append([0] * len(map[0]))
    while len(visited) != len(houses):
        source = random.choice(visited)
        mindist = len(map) * len(map[0])
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
                    g = val.g + cost(node, next, map, road)
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

        visited.append(destination)

    return road
        

def SPATH_st(map):
    houses = find_houses(map)
    visited = []
    source = random.choice(houses)
    visited.append(source)

    road = []
    for row in range(len(map)):
            road.append([0] * len(map[0]))
    for destination in houses:
        if destination == source:
            continue
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
                    g = val.g + cost(node, next, map, road)
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

        visited.append(destination)

    return road


def find(x, parent):
    if parent[x[0]][x[1]] != x:
        return find(parent[x[0]][x[1]], parent)
    return x

def union(x, y, parent):
    y_par = find(y, parent)
    parent[y_par[0]][y_par[1]] = find(x, parent)
    return parent

def fullyconnected(houses, parent):
    flag = True
    p = find(houses[0], parent)
    for house in houses:
        flag = flag and (p == find(house, parent))
    return flag

def f_v(v,components, map):
    closestpoints = []
    for component in components:
        mindist = len(map) * len(map[0])
        for u in component:
            dist = distance(v, u)
            if dist < mindist:
                mindist = dist
                closest = u
        closestpoints.append(closest)

    minavg = len(map) * len(map[0])
    for t in range(len(components)):
        sum = 0
        for i in range(t+1):
            sum += distance(v, closestpoints[i])
        sum = sum/(t+1)
        if sum < minavg:
            minavg = sum
    
    return minavg, closestpoints

def heuristic(test_v, components, map):
    min = len(map) * len(map[0])
    for v in test_v:
        f, closestpoints = f_v(v, components, map)
        if f < min:
            min = f
            best = v
    return best, closestpoints


def HEUM_st(map):
    houses = find_houses(map)
    components = []
    for house in houses:
        components.append([house])

    parent = []
    for row in range(len(map)):
        parent.append([])
        for col in range(len(map[0])):
            parent[row].append((row,col))

    road = []
    for row in range(len(map)):
            road.append([0] * len(map[0]))

    while not fullyconnected(houses, parent):
        test_v = []
        while len(test_v) < 2*len(map):
            v = (random.randint(0, len(map)-1), random.randint(0, len(map[0])-1))
            if valid(v, v, map) and road[v[0]][v[1]] != 1:
                test_v.append(v)
        v, closestpoints = heuristic(test_v, components, map)
        mindist = len(map) * len(map[0])
        for i in range(2):
            mindist = len(map) * len(map[0])
            for j in range(len(closestpoints)):
                dist = distance(v, closestpoints[j])
                if dist < mindist:
                    if i == 0:
                        mindist = dist
                        point1 = closestpoints[j]
                        index1 = j
                    elif j != index1:
                        mindist = dist
                        point2 = closestpoints[j]
                        index2 = j
        backtrack = []
        for row in range(len(map)):
            backtrack.append([(0,0)] * len(map[0]))
        closed = set()
        open = PriorityQueue(order=min, f=lambda v: v.f)
        open.put(point1, Value(distance(point1, v), 0))
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        while len(open) > 0:
            node, val = open.pop()
            closed.add(node)
            if node == v:
                break
            for move in moves:
                next = (move[0] + node[0], move[1] + node[1])
                if next not in closed and valid(next, node, map):
                    g = val.g + cost(node, next, map, road)
                    f = g + distance(next, v) 
                    open.put(next, Value(f, g))
                    backtrack[next[0]][next[1]] = node
        
        path = [v]
        curr = v
        while curr != point1:
            curr = backtrack[curr[0]][curr[1]]
            path = [curr] + path

        for tile in path:
            parent = union(point1, tile, parent)
            if tile not in components[index1]:
                components[index1].append(tile)
            road[tile[0]][tile[1]] = 1

        
        backtrack = []
        for row in range(len(map)):
            backtrack.append([(0,0)] * len(map[0]))
        closed = set()
        open = PriorityQueue(order=min, f=lambda v: v.f)
        open.put(point2, Value(distance(point2, v), 0))
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        while len(open) > 0:
            node, val = open.pop()
            closed.add(node)
            if node == v:
                break
            for move in moves:
                next = (move[0] + node[0], move[1] + node[1])
                if next not in closed and valid(next, node, map):
                    g = val.g + cost(node, next, map, road)
                    f = g + distance(next, v) 
                    open.put(next, Value(f, g))
                    backtrack[next[0]][next[1]] = node
        
        path = [v]
        curr = v
        while curr != point2:
            curr = backtrack[curr[0]][curr[1]]
            path = [curr] + path

        for tile in path:
            parent = union(point2, tile, parent)
            if tile not in components[index1] and tile not in components[index2]:
                components[index2].append(tile)
            road[tile[0]][tile[1]] = 1

        parent = union(point1, point2, parent)
        components[index1].extend(components[index2])
        del components[index2]

    return road

def eval(map, method, iterations=250):
    lengths = []
    times = []
    minlength = len(map) * len(map[0])
    for i in range(iterations):
        start = time.time()
        road = method(map)
        times.append(time.time()-start)
        sum = 0
        for row in road:
            for item in row:
                if item == 1:
                    sum += 1
        lengths.append(sum)
        if sum < minlength:
            minlength = sum
    avgoptimalitygap = 0
    avglength = 0
    avgtime = 0
    for i in lengths:
        avgoptimalitygap += i-minlength
        avglength += i
    for i in times:
        avgtime += i
    avgoptimalitygap /= len(lengths)
    avglength /= len(lengths)
    avgtime /= len(times)
    return avgoptimalitygap, avglength, avgtime
        



    
map = parse_map('giantmap.txt')
print(eval(map, HEUM_st))
'''
road = SPATH_st(map)
print(map)
print(np.matrix(road))'''

        
    
