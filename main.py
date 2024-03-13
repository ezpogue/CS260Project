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

map = parse_map('testmap.txt')
print(find_houses(map))

