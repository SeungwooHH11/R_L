import numpy as np
from collections import deque

device = 'cuda'
import torch


# Backward DP
def backtracking_dp(grid, goal):
    n, m = len(grid), len(grid[0])
    dp = np.full((n, m), -float('inf'))  # Initialize DP table with infinity
    dp[goal[0]][goal[1]] = 1
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < m and grid[x][y] != -2

    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in movements:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                step_cost = -0.1 if grid[nx][ny] == -1 else -0.001
                new_cost = dp[x][y] + step_cost
                if new_cost > dp[nx][ny]:
                    dp[nx][ny] = new_cost
                    queue.append((nx, ny))
    if np.all(dp[0] == -float('inf')):
        return False, dp
    return True, dp


def label_connected_paths(input_grid):
    # 입구에 있는 Free space들은 연결 시켜서
    grid = input_grid.copy()
    rows, cols = len(grid), len(grid[0])
    temp_grid = np.zeros((rows + 1, cols))
    temp_grid[1:, :] = grid
    grid = temp_grid
    visited = [[False for _ in range(cols)] for _ in range(rows + 1)]
    label = 2
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    label_num = np.zeros(rows * cols + rows)

    def dfs(x, y, label):
        visited[x][y] = True
        grid[x][y] = label
        if x != 0:
            label_num[label] += 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows + 1 and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 0:
                dfs(nx, ny, label)

    for i in range(rows + 1):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                dfs(i, j, label)
                label += 1
    grid = grid[1:, :]
    return grid, label_num


def backtracking_dp_with_free_space(grid, goal, labeled_grid, label_num):
    n, m = len(grid), len(grid[0])
    dp = np.full((n, m), -float('inf'))  # Initialize DP table with infinity
    free_space = np.full((n, m), -float('inf'))
    step_space = np.full((n, m), np.inf)

    dp[goal[0]][goal[1]] = 1
    step_space[goal[0], goal[1]] = 0
    free_space[goal[0], goal[1]] = 0

    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    isvisited = []

    def is_valid1(x, y):
        return 0 <= x < n and 0 <= y < m and grid[nx][ny] != -2 and dp[nx][ny] != 1

    def is_valid2(x, y):
        return 0 <= x < n and 0 <= y < m and grid[nx][ny] != -2

    matrix = [[[] for _ in range(m)] for _ in range(n)]
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()

        for dx, dy in movements:

            nx, ny = x + dx, y + dy
            if is_valid1(nx, ny):
                # 원점 배재
                temp = np.ones(4) * -100
                e = 0
                for ndx, ndy in movements:

                    if is_valid2(nx + ndx, ny + ndy):
                        # 원점 포함해서 dp value 서치
                        temp[e] = dp[nx + ndx, ny + ndy]
                    e += 1
                ndx, ndy = movements[np.argmax(temp)]
                step_space[nx][ny] = step_space[nx + ndx][ny + ndy] + 1
                free_space[nx][ny] = free_space[nx + ndx][ny + ndy]

                if grid[nx][ny] == 0:
                    matrix[nx][ny] = matrix[nx + ndx][ny + ndy].copy()

                    if labeled_grid[nx, ny] not in matrix[nx + ndx][ny + ndy]:
                        free_space[nx, ny] += label_num[int(labeled_grid[nx, ny])]
                        matrix[nx][ny].append(labeled_grid[nx, ny].copy())
                    new_cost = dp[nx + ndx][ny + ndy] - 0.001
                if grid[nx][ny] == -1:
                    if step_space[nx, ny] > free_space[nx, ny]:
                        new_cost = dp[nx + ndx][ny + ndy] - 0.2
                    else:
                        new_cost = dp[nx + ndx][ny + ndy] - 0.1
                if new_cost > dp[nx, ny]:
                    dp[nx, ny] = new_cost
                    queue.append((nx, ny))

    if np.all(dp[0] == -float('inf')):
        return False, dp, 0, 0
    return True, dp, free_space, step_space


def search_path(result, grid):
    start = [0, np.argmax(result[0, :])]

    rearrange_count = 9 - int(result[start[0], start[1]] * 10)
    if result[start[0], start[1]] * 100 % 10 == 0:
        rearrange_count += 1
    current = start
    path = [current.copy()]
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while result[current[0], current[1]] != 1:
        temp = np.ones(4) * -100
        step = 0

        def is_valid(x, y):
            return 0 <= x < result.shape[0] and 0 <= y < result.shape[1]

        for dx, dy in (movements):
            nx, ny = current[0] + dx, current[1] + dy
            if is_valid(nx, ny):
                temp[step] = result[nx, ny]
            step += 1
        dx, dy = movements[np.argmax(temp)]
        current[0] = current[0] + dx
        current[1] = current[1] + dy
        if result[current[0], current[1]] != 1:
            path.append(current.copy())

    return path, rearrange_count


def cal_area(path, labeled_grid, label_num, input_dp, grid):
    dp = input_dp.copy()
    area_required = 0
    area_able = 0
    path_label = []
    added_label = []
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    area_left = 0
    before_label = 1
    for i in path:
        if grid[i[0], i[1]] == -1:
            area_required += 2
            area_able += 1
            for dx, dy in movements:
                if 0 <= i[0] + dx < labeled_grid.shape[0] and 0 <= i[1] + dy < labeled_grid.shape[1]:
                    if [i[0] + dx, i[1] + dy] not in path and labeled_grid[i[0] + dx, i[1] + dy] > 1:
                        if labeled_grid[i[0] + dx, i[1] + dy] not in added_label:
                            added_label.append(labeled_grid[i[0] + dx, i[1] + dy])

                            area_able += label_num[int(labeled_grid[i[0] + dx, i[1] + dy])]

        elif grid[i[0], i[1]] == 0:

            area_required += 1
            if labeled_grid[i[0], i[1]] not in path_label:
                before_label = labeled_grid[i[0], i[1]]
                path_label.append(labeled_grid[i[0], i[1]])
                area_able += label_num[int(labeled_grid[i[0], i[1]])]
    area_left = area_able - area_required
    return area_able, area_required, area_left, dp, path_label, added_label


def path_finder(grid, goal):
    labeled_grid, label_num = label_connected_paths(grid.copy())
    ispossible, dp, _, _ = backtracking_dp_with_free_space(grid.copy(), goal, labeled_grid, label_num)
    if not ispossible:
        return False, 0, 0, 0, 0, 0, 0, 0
    path, rearrange_count = search_path(dp, grid)
    area_able, area_required, area_left, new_dp, path_label, added_label = cal_area(path, labeled_grid, label_num, dp,
                                                                                    grid.copy())
    return True, path, area_left, rearrange_count, path_label, added_label, labeled_grid, label_num


def bfs_area(final_grid, grid, start, path):  # visited, gird, goal
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우 이동
    queue = deque([tuple(start)])
    vs = set([tuple(start)])
    area_size = 0

    obstacle = []
    free_space = []

    while queue:
        r, c = queue.popleft()

        area_size += 1  # 현재 위치 포함
        if grid[r][c] == -1:
            obstacle.append([r, c])
        if grid[r][c] == 0 and [r, c] not in path:
            free_space.append([r, c])
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in vs and final_grid[nr][nc] == 1:
                vs.add((nr, nc))
                queue.append((nr, nc))

    return obstacle, free_space


def Create_locate_mask(grid, TP_type_len):
    r, c, f = grid.shape
    mask = (grid[:, :, 1:1 + TP_type_len].sum(axis=2) > 0).astype(np.uint8)  # 첫 번째 열이 0 초과인 위치를 1로 설정
    mask = mask[:, :, np.newaxis].copy()
    mask = mask.reshape(r, c)
    rows, cols = len(mask), len(mask[0])
    visited = [[False] * cols for _ in range(rows)]  # 방문 여부 기록
    new_grid = [[1] * cols for _ in range(rows)]  # 모든 값을 1로 초기화

    # BFS를 위한 큐
    queue = deque()

    # Step 1: 첫 번째 행에서 0을 찾고 BFS 시작
    for x in range(cols):
        if mask[0][x] == 0:
            queue.append((0, x))
            visited[0][x] = True  # 방문 체크
            new_grid[0][x] = 0  # 그대로 유지

    # BFS 탐색 (상, 하, 좌, 우)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()

        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            if 0 <= ny < rows and 0 <= nx < cols and not visited[ny][nx] and mask[ny][nx] == 0:
                queue.append((ny, nx))
                visited[ny][nx] = True
                new_grid[ny][nx] = 0  # 유지
    new_grid = np.array(new_grid)

    return new_grid


def bfs_path_exists(grid, start, goal):
    start = start.copy()
    goal = goal.copy()
    start[0] = start[0] + 1
    goal[0] = goal[0] + 1
    rows, cols = grid.shape  # numpy 배열 크기 가져오기
    temp_grid = np.zeros((rows + 1, cols))
    temp_grid[1:, :] = grid
    grid = temp_grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우 이동
    queue = deque([tuple(start)])
    vs = set([tuple(start)])
    while queue:

        r, c = queue.popleft()
        if r == goal[0] and c == goal[1]:
            return True  # 목표 지점 도달
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in vs and grid[nr][nc] == 0:
                vs.add((nr, nc))
                queue.append((nr, nc))

    return False  # 목표 지점에 도달할 수 없음


def Create_mask(grid, TP_type_len):
    mask = (grid[:, :, 1:1 + TP_type_len].sum(axis=2) > 0).astype(np.uint8)  # 첫 번째 열이 0 초과인 위치를 1로 설정
    return mask[:, :, np.newaxis]


def retrieval(final_grid, input_grid, state_grid, target_block, path, ppo, step, grids, blocks, block_lefts,
              block_left_num, rewards, actions, dones, masks, probs, lookahead_num, TP_type_len, rt_mod):
    do = 0
    stock = []
    obstacles_set = []
    for st in path:
        if input_grid[st[0], st[1]] == -1:
            obstacles_set.append([st[0], st[1]])

    rearrange_num = len(obstacles_set)
    goal = target_block
    for ob_num in range(rearrange_num):
        obstacles, free_spaces = bfs_area(final_grid, input_grid, goal, path)
        target_obstacle = obstacles_set[ob_num]
        possible_space = []
        for free_space in free_spaces:
            if bfs_path_exists(input_grid, target_obstacle, free_space):
                possible_space.append(free_space)
        if len(possible_space) == 0:
            stock.append(state_grid[target_obstacle[0], target_obstacle[1], :-1].copy())

            input_grid[target_obstacle[0]][target_obstacle[1]] = 0
            change_reward = state_grid[target_obstacle[0], target_obstacle[1], -1]

            rewards[change_reward] -= 0.2
            state_grid[target_obstacle[0], target_obstacle[1]] = 0
            # print('Obstacle at ',target_obstacle,'temporary retrieved')
        else:
            obstacle_left = rearrange_num - ob_num  # 총 2- 0 = 2
            can = []
            for free_space in possible_space:
                test_grid1 = final_grid.copy()
                test_grid1[free_space[0]][free_space[1]] = 0
                test_grid1[target_obstacle[0]][target_obstacle[1]] = 1
                test_grid2 = input_grid.copy()
                test_grid2[target_obstacle[0]][target_obstacle[1]] = 0
                test_grid2[free_space[0]][free_space[1]] = -1
                temp_obstacle, temp_free_space = bfs_area(test_grid1, test_grid2, goal, path)
                can.append(min(len(temp_free_space) - len(temp_obstacle), 0))
            max_value = max(can)  # 최대 값 찾기
            max_indices = [index for index, value in enumerate(can) if value == max_value]
            # maks 1, r*c, 1
            mask = np.ones((input_grid.shape[0], input_grid.shape[1]))
            for ind in max_indices:
                mask[possible_space[ind][0], possible_space[ind][1]] = 0
            blocks_vec = np.zeros((lookahead_num, TP_type_len + 1))
            blocks_vec[:, 0] = 250
            blocks_vec[:, 1:1 + int((1 + TP_type_len) / 2.0)] = 1
            blocks_vec[0] = state_grid[target_obstacle[0], target_obstacle[1], :-1]
            grids.append(state_grid.copy())
            blocks.append(blocks_vec.copy())
            block_lefts.append(block_left_num)
            masks.append(mask.reshape(-1, 1).copy())
            mask = torch.tensor(mask.reshape(1, -1, 1), dtype=torch.float32).to(device)
            state_grid_tensor = torch.tensor(
                state_grid[:, :, :-1].reshape(1, state_grid.shape[0], state_grid.shape[1], -1), dtype=torch.float32).to(
                device)
            state_grid_tensor[:, :, 0] = state_grid_tensor[:, :, 0] / 500.0
            blocks_vec_tensor = torch.tensor(blocks_vec.reshape(1, lookahead_num, -1), dtype=torch.float32).to(device)
            blocks_vec_tensor[:, :, 0] = blocks_vec_tensor[:, :, 0] / 500.0
            pr, target_space = ppo.Locate(state_grid_tensor, blocks_vec_tensor, mask, ans=None)
            target_r = target_space.item() // input_grid.shape[0]
            target_c = target_space.item() % input_grid.shape[1]
            probs.append(pr.item())
            actions.append(target_space.item())
            dones.append(0)
            rewards.append(0)
            step += 1
            change_reward = state_grid[target_obstacle[0], target_obstacle[1], -1]

            rewards[change_reward] -= 0.1
            state_grid[target_r, target_c] = state_grid[target_obstacle[0], target_obstacle[1]].copy()
            state_grid[target_r, target_c, -1] = step
            state_grid[target_obstacle[0], target_obstacle[1]] = 0

            final_grid[target_r][target_c] = 0
            input_grid[target_r][target_c] = -1
            input_grid[target_obstacle[0]][target_obstacle[1]] = 0
            # print('Obstacle at ',target_obstacle)
            # print('to', target_space)
    if rt_mod == 'OR':
        state_grid[goal[0], goal[1], :] = 0
        final_grid[goal[0], goal[0]] = 1
        input_grid[goal[0], goal[1]] = 0
    for e, ob in enumerate(stock):
        can = []
        mask = Create_locate_mask(state_grid, TP_type_len)  # r,c
        zero_positions = np.argwhere(mask == 0)
        obstacle_left = len(stock) - e - 1
        for row_num, col_num in zero_positions:
            temp_grid = state_grid.copy()

            temp_grid[row_num, col_num, :-1] = ob.copy()
            mask = Create_locate_mask(temp_grid, TP_type_len)  # r,c
            new_zero_positions = np.argwhere(mask == 0)
            can.append(min(len(new_zero_positions) - obstacle_left, 0))
        max_value = max(can)  # 최대 값 찾기
        max_indices = [index for index, value in enumerate(can) if value == max_value]
        # maks 1, r*c, 1
        final_mask = np.ones((input_grid.shape[0], input_grid.shape[1]))
        for ind in max_indices:
            final_mask[zero_positions[ind][0], zero_positions[ind][1]] = 0
        blocks_vec = np.zeros((lookahead_num, TP_type_len + 1))
        blocks_vec[:, 0] = 250
        blocks_vec[:, 1:1 + int((1 + TP_type_len) / 2.0)] = 1
        max_copy = min(len(stock) - e, lookahead_num)
        blocks_vec[:max_copy] = np.array(stock[e:e + max_copy])
        block_lefts.append(block_left_num)
        grids.append(state_grid.copy())
        blocks.append(blocks_vec.copy())
        masks.append(final_mask.reshape(-1, 1).copy())
        final_mask = torch.tensor(final_mask.reshape(1, -1, 1), dtype=torch.float32).to(device)
        state_grid_tensor = torch.tensor(state_grid[:, :, :-1].reshape(1, state_grid.shape[0], state_grid.shape[1], -1),
                                         dtype=torch.float32).to(device)
        state_grid_tensor[:, :, 0] = state_grid_tensor[:, :, 0] / 500.0
        blocks_vec_tensor = torch.tensor(blocks_vec.reshape(1, lookahead_num, -1), dtype=torch.float32).to(device)
        blocks_vec_tensor[:, :, 0] = blocks_vec_tensor[:, :, 0] / 500.0

        pr, target_space = ppo.Locate(state_grid_tensor, blocks_vec_tensor, final_mask, ans=None)

        target_r = target_space.item() // input_grid.shape[0]
        target_c = target_space.item() % input_grid.shape[1]
        probs.append(pr.item())
        actions.append(target_space.item())
        dones.append(0)
        rewards.append(0)
        step += 1
        state_grid[target_r, target_c, :-1] = ob
        state_grid[target_r, target_c, -1] = step

        final_grid[target_r][target_c] = 0
        input_grid[target_r][target_c] = -1
        # print('Retrieved obstacle')
        # print('to', target_space)

    return rearrange_num, state_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts


def classify_grid(grid, TP_capacity, goal):  #
    M = TP_capacity
    r = grid.shape[0]
    c = grid.shape[1]
    classified_grid = np.zeros((r, c), dtype=int)

    for i in range(r):
        for j in range(c):
            if grid[i, j, 1 + M] == 1:
                classified_grid[i, j] = -1
            elif grid[i, j, 1 + M:].sum() > 0:
                classified_grid[i, j] = -2
            else:
                classified_grid[i, j] = 0
    classified_grid[goal[0]][goal[1]] = 1

    return classified_grid


def Retrieval(grid, TP_capacity, target_block, ppo, step, grids, blocks, block_lefts, block_left_num, actions, rewards,
              dones, masks, probs, lookahead_num, TP_type_len, rt_mod):
    # grid r,c, 2+TP_type
    input_grid = classify_grid(grid, TP_capacity, target_block)
    ispossible, path, area_left, count, path_label, added_label, labeled_grid, label_num = path_finder(
        input_grid.copy(), target_block)
    if ispossible == False:
        rearrange_num = 0
        return ispossible, rearrange_num, grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts
    total_area = []
    for label in added_label:
        total_area = list(np.array(np.where(labeled_grid == label)).T) + total_area
    for label in path_label:
        total_area = list(np.array(np.where(labeled_grid == label)).T) + total_area
    total_area = total_area + path
    final_grid = np.zeros((grid.shape[0], grid.shape[1]))

    for x, y in total_area:
        final_grid[x][y] = 1  # 초기 확정 area
    # final_grid=space_finder(input_grid,path,area_left,count,path_label,labeled_grid,label_num,added_label)

    rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = retrieval(
        final_grid, input_grid, grid.copy(), target_block, path, ppo, step, grids, blocks, block_lefts, block_left_num,
        rewards, actions, dones, masks, probs, lookahead_num, TP_type_len, rt_mod)

    return ispossible, rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts


def Count_retrieval(grid, TP_capacity, target_block):
    input_grid = classify_grid(grid, TP_capacity, target_block)
    ispossible, path, area_left, count, path_label, added_label, labeled_grid, label_num = path_finder(
        input_grid.copy(), target_block)
    return count, ispossible

