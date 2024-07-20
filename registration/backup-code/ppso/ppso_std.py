import numpy as np

def griewank(x):
    xs = x ** 2
    sum_term = np.sum(xs, axis=1) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
    return sum_term - prod_term + 1

def ppso(layers, dimensions, lower_bound, upper_bound, max_fes, func_id):
    num_layers = len(layers)
    cumulative_layers = np.cumsum(layers)
    total_particles = np.sum(layers)
    fes = 0
    fitness = np.full(total_particles, 1e200)

    if dimensions == 50:
        phi = 0.04
    elif dimensions == 30:
        phi = 0.02
    else:
        phi = 0.008

    # 用来记录粒子的对应的层级关系，这样能够保证每个层级的粒子都能保存进去，尽管不会填充完
    particle_positions = np.zeros((num_layers, max(layers), dimensions))
    particle_velocities = np.zeros((num_layers, max(layers), dimensions))
    fitness_per_layer = np.zeros((num_layers, max(layers), 1))
    personal_best_positions = np.zeros((num_layers, max(layers), dimensions))

    position_min = np.tile(lower_bound, (total_particles, 1))
    position_max = np.tile(upper_bound, (total_particles, 1))

    positions = position_min + (position_max - position_min) * np.random.rand(total_particles, dimensions)
    velocities = 0.1 * (position_min + (position_max - position_min) * np.random.rand(total_particles, dimensions))
    personal_best_positions_all = positions.copy()
    global_best_fitness = 1e200

    if func_id == 1:
        fitness_values = griewank(positions)
    else:
        raise ValueError("Unsupported func_id")

    fes = total_particles
    best_ever_fitness = global_best_fitness
    fitness = fitness_values
    max_generations = max_fes // total_particles
    # 用于记录历史最优
    best_fitness_history = np.zeros(max_generations)

    generation = 1
    temp_positions = np.zeros((total_particles, dimensions))
    temp_velocities = np.zeros((total_particles, dimensions))
    temp_personal_best_positions = np.zeros((total_particles, dimensions))
    temp_fitness = 9999999 * np.ones(total_particles)
    layer_indices = np.zeros((num_layers, 2))
    all_positions = np.zeros((total_particles, dimensions))
    all_velocities = np.zeros((total_particles, dimensions))

    while fes < max_fes:
        sorted_fitness = np.sort(fitness)
        # 该函数返回的是数组值适应值从小到大排序时的索引值。
        sorted_indices = np.argsort(fitness)

        best_fitness_history[generation - 1] = sorted_fitness[0]
        
        # 顶层金字塔的索引
        indices = sorted_indices[:cumulative_layers[0]]
        # 得到顶层金字塔粒子的位置以及速度
        particle_positions[0, :len(indices), :] = positions[indices, :]
        particle_velocities[0, :len(indices), :] = velocities[indices, :]
        # 索引切片之后再进行升维
        fitness_per_layer[0, :len(indices), :] = fitness[indices, np.newaxis]
        # 金字塔中保留个体最优的粒子
        personal_best_positions[0, :len(indices), :] = personal_best_positions_all[indices, :]
        # 从剩下的金字塔层中开始查找，将每一层的粒子排好序放入对应的金字塔数据结构中
        for layer_index in range(1, num_layers):
            indices = sorted_indices[cumulative_layers[layer_index - 1]:cumulative_layers[layer_index]]
            len_indices = len(indices)
            particle_positions[layer_index, :len_indices, :] = positions[indices, :]
            particle_velocities[layer_index, :len_indices, :] = velocities[indices, :]
            fitness_per_layer[layer_index, :len_indices, :] = fitness[indices, np.newaxis]
            personal_best_positions[layer_index, :len_indices, :] = personal_best_positions_all[indices, :]

        temp_index = 0
        losers_list = []
        winners_list = []
        
        # 开始更新速度和位置，逐层更新，至下而上
        for layer_index in range(num_layers - 1, -1, -1):
            layer_size = layers[layer_index]
            random_indices = np.random.permutation(layer_size) # 生成一个随机排序的序列
            separator = layer_size // 2
            random_pairs = np.column_stack((random_indices[:separator], random_indices[separator:2 * separator]))
            
            comparison_mask = (fitness_per_layer[layer_index, random_pairs[:, 0], 0] > fitness_per_layer[layer_index, random_pairs[:, 1], 0])
            losers = np.where(comparison_mask, random_pairs[:, 0], random_pairs[:, 1])
            winners = np.where(~comparison_mask, random_pairs[:, 0], random_pairs[:, 1])

            random_coeff1 = np.random.rand(separator, dimensions)
            random_coeff2 = np.random.rand(separator, dimensions)
            random_coeff3 = np.random.rand(separator, dimensions)
            
            loser_velocities = particle_velocities[layer_index, losers, :].reshape(separator, dimensions)
            loser_positions = particle_positions[layer_index, losers, :].reshape(separator, dimensions)
            loser_personal_bests = personal_best_positions[layer_index, losers, :].reshape(separator, dimensions)
            
            winner_velocities = particle_velocities[layer_index, winners, :].reshape(separator, dimensions)
            winner_positions = particle_positions[layer_index, winners, :].reshape(separator, dimensions)
            winner_personal_bests = personal_best_positions[layer_index, winners, :].reshape(separator, dimensions)
            
            top_layer_size = layers[0]
            # 不足那么多个粒子时就随意组合
            top_indices = np.random.permutation(separator) % top_layer_size
            global_best_positions = particle_positions[0, top_indices, :].reshape(separator, dimensions)
            
            # loser的速度和位置更新
            updated_loser_velocities = (random_coeff1 * loser_velocities +
                                        random_coeff2 * (winner_positions - loser_positions) +
                                        random_coeff3 * (loser_personal_bests - loser_positions))
            updated_loser_positions = loser_positions + updated_loser_velocities

            if layer_index != 0:
                # 非顶层winner更新速度和位置
                # 获取上层的速度和位置，向更好的winner学习
                upper_layer_size = layers[layer_index - 1]
                upper_indices = np.random.permutation(separator) % upper_layer_size
                upper_positions = particle_positions[layer_index - 1, upper_indices, :].reshape(separator, dimensions)

                random_coeff4 = np.random.rand(separator, dimensions)
                # 这里的global_best_positions是从顶层得到的，此时会存在粒子不足的问题，
                # 比如这层有16个，顶层有4个，解决方案是将顶层的这几个复制在打乱
                updated_winner_velocities = (random_coeff1 * winner_velocities +
                                             random_coeff2 * (upper_positions - winner_positions) +
                                             random_coeff3 * (winner_personal_bests - winner_positions) +
                                             phi * random_coeff4 * (global_best_positions - winner_positions))
                updated_winner_positions = winner_positions + updated_winner_velocities
            else:
                # 顶层winner更新速度和位置
                updated_winner_velocities = winner_velocities
                updated_winner_positions = winner_positions

            merged_positions = np.vstack((updated_loser_positions, updated_winner_positions))
            num_losers = len(updated_loser_positions)
            num_winners = len(updated_winner_positions)
            layer_indices[layer_index, :] = [num_losers, num_winners]

            merged_velocities = np.vstack((updated_loser_velocities, updated_winner_velocities))
            all_positions[temp_index:temp_index + num_losers + num_winners, :] = merged_positions
            all_velocities[temp_index:temp_index + num_losers + num_winners, :] = merged_velocities
            temp_index += num_losers + num_winners
            # 后面是倒着来的
            losers_list.insert(0, losers)
            winners_list.insert(0,winners)

        # 限定粒子参数范围
        all_positions = np.clip(all_positions, lower_bound, upper_bound)
        if func_id == 1:
            new_fitness_values = griewank(all_positions)
        else:
            raise ValueError("Unsupported func_id")

        # 这是一种并行化的处理方式
        temp_index = 0
        for layer_index in range(num_layers - 1, -1, -1):
            num_losers, num_winners = layer_indices[layer_index]
            num_losers = int(num_losers)
            num_winners = int(num_winners)
            loser_fitness_values = new_fitness_values[temp_index:temp_index + num_losers]
            loser_positions = all_positions[temp_index:temp_index + num_losers]
            loser_velocities = all_velocities[temp_index:temp_index + num_losers]
            winner_fitness_values = new_fitness_values[temp_index + num_losers:temp_index + num_losers + num_winners]
            winner_positions = all_positions[temp_index + num_losers:temp_index + num_losers + num_winners]
            winner_velocities = all_velocities[temp_index + num_losers:temp_index + num_losers + num_winners]
            temp_index += num_losers + num_winners

            losers = losers_list[layer_index]
            winners = winners_list[layer_index]

            good_loser_indices = fitness_per_layer[layer_index, losers, 0] > loser_fitness_values
            good_winner_indices = fitness_per_layer[layer_index, winners, 0] > winner_fitness_values
            good_losers = losers[good_loser_indices]
            good_winners = winners[good_winner_indices]

            personal_best_positions[layer_index, good_losers, :] = loser_positions[good_loser_indices]
            personal_best_positions[layer_index, good_winners, :] = winner_positions[good_winner_indices]

            particle_velocities[layer_index, losers, :] = loser_velocities
            particle_positions[layer_index, losers, :] = loser_positions

            particle_velocities[layer_index, winners, :] = winner_velocities
            particle_positions[layer_index, winners, :] = winner_positions

            fitness_per_layer[layer_index, losers, 0] = loser_fitness_values
            fitness_per_layer[layer_index, winners, 0] = winner_fitness_values

        for layer_index in range(num_layers):
            if layer_index == 0:
                left_bound = 0
                right_bound = cumulative_layers[layer_index]
            else:
                left_bound = cumulative_layers[layer_index - 1]
                right_bound = cumulative_layers[layer_index]

            temp_positions[left_bound:right_bound, :] = particle_positions[layer_index, :layers[layer_index]].reshape(layers[layer_index], dimensions)
            temp_velocities[left_bound:right_bound, :] = particle_velocities[layer_index, :layers[layer_index]].reshape(layers[layer_index], dimensions)
            temp_personal_best_positions[left_bound:right_bound, :] = personal_best_positions[layer_index, :layers[layer_index]].reshape(layers[layer_index], dimensions)
            temp_fitness[left_bound:right_bound] = fitness_per_layer[layer_index, :layers[layer_index], 0]

        positions = temp_positions
        velocities = temp_velocities
        personal_best_positions_all = temp_personal_best_positions

        fitness = temp_fitness
        best_ever_fitness = min(best_ever_fitness, fitness.min())
        fes += total_particles
        generation += 1

    return best_ever_fitness, fitness, positions, best_fitness_history

def main():
    np.random.seed(74)
    runs = 10
    layers = [4, 8, 20, 32]
    dimensions = 30
    max_fes = dimensions * 10000
    func_id = 1
    lower_bound = -600
    upper_bound = 600
    results = np.ones(runs) * 99999999999

    for run_index in range(runs):
        results[run_index], fitness, positions, best_fitness_history = ppso(layers, dimensions, lower_bound, upper_bound, max_fes, func_id)
        print(f'{run_index + 1} : {results[run_index]:e}')
    
    print('\n\n====================\n\n')
    print(f'FID:{func_id} mean result: {np.mean(results):e}')

if __name__ == "__main__":
    main()
