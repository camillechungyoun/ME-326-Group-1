import numpy as np
import itertools
from itertools import product
import pprint


class OccupationGrid:
    def __init__(self, map_size=(6.0, 6.0), step_size=.05):
        self._step_size = step_size  # meters
        grid_size = np.array(map_size) / step_size

        self._red_grid = np.zeros(grid_size.astype(int))
        self._blue_grid = np.zeros(grid_size.astype(int))
        self._green_grid = np.zeros(grid_size.astype(int))
        self._yellow_grid = np.zeros(grid_size.astype(int))

        self.l_occ = np.log(0.55 / 0.45)
        self.l_free = np.log(0.45 / 0.55)

    def update(self, x, y, color=None):
        assert color in ['red', 'blue', 'green', 'yellow']

        x_snap = np.round(x / self._step_size).astype(int)
        y_snap = np.round(y / self._step_size).astype(int)

        mask = np.zeros_like(self._red_grid, dtype=bool)
        mask[x_snap, y_snap] = True

        if color == 'red':
            self._red_grid[mask] += self.l_occ
            self._red_grid[~mask] += self.l_free
        if color == 'blue':
            self._blue_grid[mask] += self.l_occ
            self._blue_grid[~mask] += self.l_free
        if color == 'green':
            self._green_grid[mask] += self.l_occ
            self._green_grid[~mask] += self.l_free
        if color == 'yellow':
            self._yellow_grid[mask] += self.l_occ
            self._yellow_grid[~mask] += self.l_free

    def get_block_locs(self, color):
        assert color in ['red', 'blue', 'green', 'yellow']

        idxs = None
        if color == 'red':
            idxs = np.argwhere(1.0 - 1. / (1. + np.exp(self._red_grid)) > .75)
            # return idxs * self._step_size
            return np.array([[0, 1], [4, 4], [9, 5]])  # For testing the nash-equlibrium code, comment out to use the actual grid

        if color == 'blue':
            idxs = np.argwhere(1.0 - 1. / (1. + np.exp(self._red_grid)) > .75)
            # return idxs * self._step_size
            return np.array([[9, 2], [4, 0], [5, 9]])  # For testing the nash-equlibrium code, comment out to use the actual grid

        if color == 'green':
            idxs = np.argwhere(1.0 - 1. / (1. + np.exp(self._red_grid)) > .75)
            # return idxs * self._step_size
            return np.array([[0, 3], [2, 8], [3, 1]])  # For testing the nash-equlibrium code, comment out to use the actual grid

        if color == 'yellow':
            idxs = np.argwhere(1.0 - 1. / (1. + np.exp(self._red_grid)) > .75)
            # return idxs * self._step_size
            return np.array([[10, 10], [8, 2], [0, 7]])  # For testing the nash-equlibrium code, comment out to use the actual grid

        return idxs


class Env:
    def __init__(self):
        self.grid = OccupationGrid()
        self.red_block_locs = None
        self.blue_block_locs = None

        self.station_state = None
        self.block_goal_assignments = None

        self.target_block = None
        self.target_goal = None
        self.station = None
        self.block_assignment = None
        self.allowed_colors = None

        self.load()

    def load(self, file=None):
        self.stations = {}
        self.station_locs = np.array([[3, 3], [8, 8], [0, 9]])  # gets loaded
        self.station_state = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # gets loaded
        self.station_reqs = np.array([[2, 0, 0, 0], [1, 0, 0, 1]])  # gets loaded
        self.allowed_colors = ['red', 'blue']  # gets loaded

        self.col2idx = {'red': 0, 'blue': 1, 'yellow': 2, 'green': 3}
        self.num_stations = len(self.station_locs)
        self.num_station_reqs = len(self.station_reqs)

    def calc_cost_assignment(self, station_locs, station_reqs, color):
        red_block_locs = self.grid.get_block_locs('red')

        # Get dist between block and each goal
        red_block_dist = np.linalg.norm(red_block_locs[:, None] - station_locs, axis=2)  # col is station, rows are blocks

        # assign each block  with temporary index
        id = np.array([x for x in range(len(red_block_locs))]).reshape(-1, 1)
        goal_red = np.hstack((id, red_block_dist))

        cost = 0
        goal2redBlock = {}
        # now loop through the stations and assign the closest blocks which satisfy the reqs
        for i, (station_loc, station_req) in enumerate(zip(station_locs, station_reqs)):
            num_red_blocks = station_req[self.col2idx[color]]
            if num_red_blocks <= 0:
                cost += abs(num_red_blocks) * 2  # cost for moving a block out
            else:
                closest_block_idxs = goal_red[:, i + 1].argsort()
                assigned_block_row = closest_block_idxs[0:min(num_red_blocks, len(red_block_locs))]
                cost += np.sum(goal_red[assigned_block_row, i + 1])

                goal2redBlock[i] = goal_red[assigned_block_row, 0].astype(int)
                goal_red = np.delete(goal_red, assigned_block_row, axis=0)

        stationLoc2block = {}
        for key in goal2redBlock:
            stationLoc2block[str(station_locs[key])] = red_block_locs[goal2redBlock[key]]

        return stationLoc2block, cost

    def calc_nash_map(self):

        min_cost = np.inf
        station2block = {}

        for stations in itertools.combinations(zip(self.station_locs, self.station_state), self.num_station_reqs):
            for reqs in itertools.permutations(self.station_reqs):

                # Unpack and modify local station reqs depending on current state of the station
                station_locs = []
                station_reqs = []
                for i, station in enumerate(stations):
                    station_locs.append(station[0])
                    station_reqs.append(reqs[i] - station[1])
                station_locs = np.array(station_locs)
                station_reqs = np.array(station_reqs)

                # get cost of blocks and assignments
                red_blocks, red_cost = self.calc_cost_assignment(station_locs, station_reqs, 'red')
                blue_blocks, blue_cost = self.calc_cost_assignment(station_locs, station_reqs, 'blue')
                green_blocks, green_cost = self.calc_cost_assignment(station_locs, station_reqs, 'green')
                yellow_blocks, yellow_cost = self.calc_cost_assignment(station_locs, station_reqs, 'yellow')

                total_cost = red_cost + blue_cost + green_cost + yellow_cost

                if total_cost < min_cost:
                    min_cost = total_cost

                    station2block['red'] = red_blocks.copy()
                    station2block['blue'] = blue_blocks.copy()
                    station2block['green'] = green_blocks.copy()
                    station2block['yellow'] = yellow_blocks.copy()

        self.block_assignment = station2block.copy()

        return station2block, min_cost

    def query_target(self, robot_pos):

        min_dist = np.inf
        station_loc = None
        block_loc = None
        block_color = None

        for color in self.allowed_colors:
            station_dict = self.block_assignment[color]
            if station_dict:
                for key in station_dict:
                    dist = np.linalg.norm(robot_pos - station_dict[key], axis=1)
                    indx = np.argmin(dist)

                    if dist[indx] <= min_dist:
                        min_dist = dist[indx]

                        station_loc = key
                        block_loc = station_dict[key][indx]
                        block_color = color

        return station_loc, block_loc, block_color


if __name__ == '__main__':
    # grid = OccupationGrid()
    # for _ in range(50):
    #     grid.update(np.array([0,3]), np.array([0,3]),'red')
    #     # grid.update(3, 3,'red')
    #
    #     prob = 1.0 - 1./(1. + np.exp(grid._red_grid[0,0]))
    # grid.get_block_locs('red')

    env = Env()
    block_assignments, cost = env.calc_nash_map()

    print('Nash Solution')
    pprint.pprint(block_assignments)
    print('Cost:', cost)
    print('-' * 50)

    print('Query closest Station/Block to robot')
    robot_pos = np.array([9, 9])
    station_loc, block_loc, block_color = env.query_target(robot_pos)
    print('Station Loc:', station_loc, 'Block Loc:', block_loc, 'Block Col:', block_color)



