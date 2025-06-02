//something

use rand::{Error, Rng};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// Board implementation remains the same as before
#[derive(Clone, Debug)]
struct Board {
    size: usize,
    stars: HashSet<(usize, usize)>,
    regions: Vec<HashSet<(usize, usize)>>,
    stars_per_row: usize,
}

impl Board {
    fn new(size: usize, regions: Vec<HashSet<(usize, usize)>>, stars_per_row: usize) -> Self {
        Board {
            size,
            stars: HashSet::new(),
            regions,
            stars_per_row,
        }
    }

    fn print_board(&self) {
        let mut region_map = vec![vec!['.'; self.size]; self.size];

        for (i, region) in self.regions.iter().enumerate() {
            let region_char = (b'a' + i as u8) as char;
            for &(row, col) in region {
                region_map[row][col] = region_char;
            }
        }

        for row in 0..self.size {
            for col in 0..self.size {
                if self.stars.contains(&(row, col)) {
                    print!("* ");
                } else {
                    print!("{} ", region_map[row][col]);
                }
            }
            println!();
        }
    }

    fn is_valid(&self) -> bool {
        // check rows
        for row in 0..self.size {
            let stars_in_row = (0..self.size)
                .filter(|&col| self.stars.contains(&(row, col)))
                .count();
            if stars_in_row != self.stars_per_row {
                return false;
            }
        }

        // check columns
        for col in 0..self.size {
            let stars_in_col = (0..self.size)
                .filter(|&row| self.stars.contains(&(row, col)))
                .count();
            if stars_in_col != self.stars_per_row {
                return false;
            }
        }

        // check regions
        for region in &self.regions {
            let stars_in_region = region
                .iter()
                .filter(|pos| self.stars.contains(pos))
                .count();
            if stars_in_region != self.stars_per_row {
                return false;
            }
        }

        // check adjacent stars
        for &(row, col) in &self.stars {
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let new_row = row as i32 + dr;
                    let new_col = col as i32 + dc;
                    if new_row >= 0
                        && new_row < self.size as i32
                        && new_col >= 0
                        && new_col < self.size as i32
                        && self.stars.contains(&(new_row as usize, new_col as usize))
                    {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn calculate_score(&self) -> i32 {
        let mut score = 0;

        let row_col_penalty= 6;
        let region_penalty = 10;
        let adjacency_penalty = 4;
        let starcount_penalty = 10;


        //extra rule to push these algorithms towards the right solution
        score -= (self.stars.len() as i32 - (self.stars_per_row * self.regions.len()) as i32).abs() * starcount_penalty;
        if !self.is_valid() {
            score -= 20;
        }

        // count violations
        // row violations
        for row in 0..self.size {
            let stars_in_row = (0..self.size)
                .filter(|&col| self.stars.contains(&(row, col)))
                .count();
            score -= (stars_in_row as i32 - self.stars_per_row as i32).abs() * row_col_penalty;
        }

        // column violations
        for col in 0..self.size {
            let stars_in_col = (0..self.size)
                .filter(|&row| self.stars.contains(&(row, col)))
                .count();
            score -= (stars_in_col as i32 - self.stars_per_row as i32).abs() * row_col_penalty;
        }

        // region violations
        for region in &self.regions {
            let stars_in_region = region
                .iter()
                .filter(|pos| self.stars.contains(pos))
                .count();
            score -= (stars_in_region as i32 - self.stars_per_row as i32).abs() * region_penalty;
        }

        // adjacent stars violations
        for &(row, col) in &self.stars {
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let new_row = row as i32 + dr;
                    let new_col = col as i32 + dc;
                    if new_row >= 0
                        && new_row < self.size as i32
                        && new_col >= 0
                        && new_col < self.size as i32
                        && self.stars.contains(&(new_row as usize, new_col as usize))
                    {
                        score -= 1 * adjacency_penalty;
                    }
                }
            }
        }

        score
    }

    fn initialize_random(&mut self) {
        let mut rng = rand::thread_rng();
        self.stars.clear();

        while self.stars.len() < self.size * self.stars_per_row {
            let row = rng.gen_range(0..self.size);
            let col = rng.gen_range(0..self.size);
            self.stars.insert((row, col));
        }
    }

    fn init_full_stars(&mut self) {
        for i in 0..self.size {
            for j in 0..self.size {
                self.stars.insert((i, j));
            }
        }
    }
}

struct SimulatedAnnealingSolver {
    initial_temperature: f64,
    cooling_rate: f64,
    iterations_per_temperature: usize,
}

impl SimulatedAnnealingSolver {
    fn new(initial_temperature: f64, cooling_rate: f64, iterations_per_temperature: usize) -> Self {
        SimulatedAnnealingSolver {
            initial_temperature,
            cooling_rate,
            iterations_per_temperature,
        }
    }

    fn solve(&self, board: &Board) -> (Board, Duration) {
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut current_board = board.clone();
        let mut current_score = current_board.calculate_score();
        let mut temperature = self.initial_temperature;

        while temperature > 0.1 {
            for _ in 0..self.iterations_per_temperature {
                // Generate neighbor by moving a random star
                let mut neighbor = current_board.clone();
                let random_star = neighbor.stars.iter().next().cloned();
                
                if let Some(star) = random_star {
                    neighbor.stars.remove(&star);
                    let new_row = rng.gen_range(0..neighbor.size);
                    let new_col = rng.gen_range(0..neighbor.size);
                    neighbor.stars.insert((new_row, new_col));
                }

                let neighbor_score = neighbor.calculate_score();
                let score_diff = neighbor_score - current_score;

                if score_diff > 0
                    || rng.gen::<f64>() < (score_diff as f64 / temperature).exp()
                {
                    current_board = neighbor;
                    current_score = neighbor_score;
                }
            }

            temperature *= self.cooling_rate;
        }

        (current_board, start_time.elapsed())
    }
}

// aco
struct AntColonySolver {
    num_ants: usize,
    iterations: usize,
    evaporation_rate: f64,
    alpha: f64,
    beta: f64,
}

impl AntColonySolver {
    fn new(
        num_ants: usize,
        iterations: usize,
        evaporation_rate: f64,
        alpha: f64,
        beta: f64,
    ) -> Self {
        AntColonySolver {
            num_ants,
            iterations,
            evaporation_rate,
            alpha,
            beta,
        }
    }

    fn solve(&self, board: &Board) -> (Board, Duration) {
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut best_board = board.clone();
        let mut best_score = board.calculate_score();
        
        // initialize pheromone matrix
        let mut pheromones = vec![vec![1.0; board.size]; board.size];

        for iteration in 0..self.iterations {
            let mut ant_solutions = Vec::new();
            
            // For each ant
            for _ in 0..self.num_ants {
                let mut ant_board = Board::new(board.size, board.regions.clone(), board.stars_per_row);
                let target_stars = board.stars_per_row * board.regions.len(); //req stars
                let mut attempts = 0;
                let max_attempts = board.size * board.size * 2;
                
                // build solution with maximum attempts limit
                while ant_board.stars.len() < target_stars && attempts < max_attempts {
                    attempts += 1;
                    let mut candidates = Vec::new();
                    
                    // find all valid positions
                    for row in 0..board.size {
                        for col in 0..board.size {
                            if !ant_board.stars.contains(&(row, col)) {
                                let mut temp_board = ant_board.clone();
                                temp_board.stars.insert((row, col));
                                candidates.push((row, col));
                            }
                        }
                    }

                    if candidates.is_empty() {
                        break; // no valid moves available
                    }

                    // calculate selection probabilities
                    let total_pheromone: f64 = candidates
                        .iter()
                        .map(|&(r, c)| pheromones[r][c])
                        .sum();

                    if total_pheromone <= 0.0 {
                        break; // prevent division by zero
                    }

                    // select position using roulette wheel selection
                    let random_value = rng.gen::<f64>();
                    let mut cumulative_prob = 0.0;

                    for &(row, col) in &candidates {
                        let prob = pheromones[row][col] / total_pheromone;
                        cumulative_prob += prob;

                        if random_value <= cumulative_prob {
                            ant_board.stars.insert((row, col));
                            break;
                        }
                    }
                }

                ant_solutions.push((ant_board.clone(), ant_board.calculate_score()));
            }

            // update pheromones
            // evaporation
            for row in 0..board.size {
                for col in 0..board.size {
                    pheromones[row][col] *= 1.0 - self.evaporation_rate;
                    pheromones[row][col] = pheromones[row][col].max(0.1); // minimum pheromone level
                }
            }

            // add new pheromones
            for (ant_board, score) in &ant_solutions {
                let pheromone_deposit = if *score > 0 { 1.0 / (-score + 1) as f64 } else { 0.1 };
                for &(row, col) in &ant_board.stars {
                    pheromones[row][col] += pheromone_deposit;
                }
            }

            // update best solution
            if let Some((board, score)) = ant_solutions
                .iter()
                .max_by_key(|(_, score)| *score)
            {
                if *score > best_score {
                    best_board = board.clone();
                    best_score = *score;
                }
            }

            if iteration % 10 == 0 {
                //println!("Iteration {}/{}, Best score: {}", iteration, self.iterations, best_score);
            }
        }

        (best_board, start_time.elapsed())
    }
}


// for pso
#[derive(Clone)]
struct Particle {
    position: Board,
    velocity: Vec<(i32, i32)>,
    best_position: Board,
    best_score: i32,
}

impl Particle {
    fn new(board: Board) -> Self {
        let mut rng = rand::thread_rng();
        let velocity = (0..board.size * board.stars_per_row)
            .map(|_| (rng.gen_range(-1..=1), rng.gen_range(-1..=1)))
            .collect();
        
        Particle {
            position: board.clone(),
            velocity,
            best_position: board.clone(),
            best_score: board.calculate_score(),
        }
    }

    fn update_velocity(&mut self, global_best: &Board, w: f64, c1: f64, c2: f64) {
        let mut rng = rand::thread_rng();
        
        for (i, v) in self.velocity.iter_mut().enumerate() {
            let r1 = rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
            
            let cognitive = (c1 * r1) as i32;
            let social = (c2 * r2) as i32;
            
            v.0 = (w * v.0 as f64) as i32 + cognitive + social;
            v.1 = (w * v.1 as f64) as i32 + cognitive + social;
            
            v.0 = v.0.clamp(-2, 2);
            v.1 = v.1.clamp(-2, 2);
        }
    }
}

// particle swarm optimization solver
struct PSOSolver {
    num_particles: usize,
    iterations: usize,
    w: f64,  // inertia weight
    c1: f64, // cognitive parameter
    c2: f64, // social parameter
}

impl PSOSolver {
    fn new(num_particles: usize, iterations: usize, w: f64, c1: f64, c2: f64) -> Self {
        PSOSolver {
            num_particles,
            iterations,
            w,
            c1,
            c2,
        }
    }

    fn solve(&self, board: &Board) -> (Board, Duration) {
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        
        // initialize particles
        let mut particles: Vec<Particle> = (0..self.num_particles)
            .map(|_| Particle::new(board.clone()))
            .collect();
        
        let mut global_best = board.clone();
        let mut global_best_score = board.calculate_score();

        for generation in 0..self.iterations {
            for particle in &mut particles {
                // update particle position
                for (i, &(dx, dy)) in particle.velocity.iter().enumerate() {
                    if let Some(star) = &particle.position.stars.iter().nth(i).cloned() {
                        let new_x = (star.0 as i32 + dx).clamp(0, (board.size - 1) as i32) as usize;
                        let new_y = (star.1 as i32 + dy).clamp(0, (board.size - 1) as i32) as usize;
                        particle.position.stars.remove(star);
                        particle.position.stars.insert((new_x, new_y));
                    }
                }

                // update personal best
                let current_score = particle.position.calculate_score();
                if current_score > particle.best_score {
                    particle.best_score = current_score;
                    particle.best_position = particle.position.clone();
                }

                // update global best
                if current_score > global_best_score {
                    global_best_score = current_score;
                    global_best = particle.position.clone();
                }
            }

            // update velocities
            for particle in &mut particles {
                particle.update_velocity(&global_best, self.w, self.c1, self.c2);
            }
        }

        (global_best, start_time.elapsed())
    }
}

// hybrid solver combining aco and simulated annealing
struct HybridSolver {
    // ACO parameters
    num_ants: usize,
    evaporation_rate: f64,
    alpha: f64,
    beta: f64,
    
    // SA parameters
    initial_temperature: f64,
    cooling_rate: f64,
    iterations_per_temperature: usize,
}

impl HybridSolver {
    fn new(
        num_ants: usize,
        evaporation_rate: f64,
        alpha: f64,
        beta: f64,
        initial_temperature: f64,
        cooling_rate: f64,
        iterations_per_temperature: usize,
    ) -> Self {
        HybridSolver {
            num_ants,
            evaporation_rate,
            alpha,
            beta,
            initial_temperature,
            cooling_rate,
            iterations_per_temperature,
        }
    }

    fn solve(&self, board: &Board) -> (Board, Duration) {
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut best_board = board.clone();
        let mut best_score = board.calculate_score();
        
        // initialize pheromone matrix
        let mut pheromones = vec![vec![1.0; board.size]; board.size];
        let mut temperature = self.initial_temperature;
        let target_stars = board.stars_per_row * board.regions.len();
        let max_attempts = board.size * board.size * 2;
        let max_temperature_stages = 100; // Prevent infinite temperature loops
        let mut temperature_stages = 0;

        while temperature > 0.1 && temperature_stages < max_temperature_stages {
            temperature_stages += 1;
            //println!("Temperature stage {}, temp: {:.2}", temperature_stages, temperature);

            // ACO phase
            for ant in 0..self.num_ants {
                let mut ant_board = Board::new(board.size, board.regions.clone(), board.stars_per_row);
                let mut attempts = 0;
                
                // construct solution with maximum attempts limit
                while ant_board.stars.len() < target_stars && attempts < max_attempts {
                    attempts += 1;
                    let mut candidates = Vec::new();
                    
                    // find valid positions
                    for row in 0..board.size {
                        for col in 0..board.size {
                            if !ant_board.stars.contains(&(row, col)) {
                                candidates.push((row, col));
                            }
                        }
                    }

                    if candidates.is_empty() {
                        break;
                    }

                    if let Some((row, col)) = self.select_next_position(&candidates, &pheromones, &mut rng) {
                        ant_board.stars.insert((row, col));
                    } else {
                        break;
                    }
                }

                // SA phase: local search with simulated annealing
                let mut current_board = ant_board;
                let mut current_score = current_board.calculate_score();
                let mut sa_iterations = 0;
                
                for _ in 0..self.iterations_per_temperature {
                    sa_iterations += 1;
                    let mut neighbor = current_board.clone();
                    
                    // generate neighbor by moving a random star
                    if let Some(star) = neighbor.stars.iter().next().cloned() {
                        neighbor.stars.remove(&star);
                        let new_row = rng.gen_range(0..neighbor.size);
                        let new_col = rng.gen_range(0..neighbor.size);
                        neighbor.stars.insert((new_row, new_col));
                        
                        let neighbor_score = neighbor.calculate_score();
                        let score_diff = neighbor_score - current_score;

                        if score_diff > 0 || rng.gen::<f64>() < (score_diff as f64 / temperature).exp() {
                            current_board = neighbor;
                            current_score = neighbor_score;
                        }
                    }

                    // early stopping if we found a valid solution with target stars
                    if current_board.is_valid() && current_board.stars.len() == target_stars {
                        break;
                    }
                }

                // update best solution
                if current_score > best_score {
                    best_board = current_board.clone();
                    best_score = current_score;
                    //println!("New best score: {} (Ant {}, Temperature stage {})", 
                    //    best_score, ant + 1, temperature_stages);
                }
                // update pheromones
                self.update_pheromones(&mut pheromones, &current_board, current_score);
            }

            temperature *= self.cooling_rate;
        }

        (best_board, start_time.elapsed())
    }

    fn select_next_position(
        &self,
        candidates: &[(usize, usize)],
        pheromones: &[Vec<f64>],
        rng: &mut rand::rngs::ThreadRng,
    ) -> Option<(usize, usize)> {
        if candidates.is_empty() {
            return None;
        }

        let total_pheromone: f64 = candidates
            .iter()
            .map(|&(r, c)| pheromones[r][c].powf(self.alpha))
            .sum();

        if total_pheromone <= 0.0 {
            return Some(candidates[rng.gen_range(0..candidates.len())]);
        }

        let random_value = rng.gen::<f64>();
        let mut cumulative_prob = 0.0;

        for &(row, col) in candidates {
            let prob = pheromones[row][col].powf(self.alpha) / total_pheromone;
            cumulative_prob += prob;

            if random_value <= cumulative_prob {
                return Some((row, col));
            }
        }

        // fallback to random selection if no position was selected
        Some(candidates[rng.gen_range(0..candidates.len())])
    }

    fn update_pheromones(&self, pheromones: &mut [Vec<f64>], board: &Board, score: i32) {
        // evaporation
        for row in pheromones.iter_mut() {
            for pheromone in row.iter_mut() {
                *pheromone *= 1.0 - self.evaporation_rate;
                *pheromone = pheromone.max(0.1); // minimum pheromone level
            }
        }

        // add new pheromones
        let pheromone_deposit = if score > 0 { 1.0 / (-score + 1) as f64 } else { 0.1 };
        for &(row, col) in &board.stars {
            pheromones[row][col] += pheromone_deposit;
        }
    }
}

struct EvolutionarySolver {
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
}

impl EvolutionarySolver {
    fn new(population_size: usize, generations: usize, mutation_rate: f64) -> Self {
        EvolutionarySolver {
            population_size,
            generations,
            mutation_rate,
        }
    }

    fn solve(&self, board: &Board) -> (Board, Duration) {
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();

        // Initialize population
        let mut population: Vec<Board> = (0..self.population_size)
            .map(|_| {
                let mut new_board = board.clone();
                new_board.initialize_random();
                new_board
            })
            .collect();

        // Evolution loop
        for generation in 0..self.generations {
            // Evaluate fitness
            let mut fitness: Vec<(Board, i32)> = population
                .iter()
                .map(|b| (b.clone(), b.calculate_score()))
                .collect();

            // Sort by fitness (higher is better)
            fitness.sort_by_key(|&(_, score)| -score);

            // Selection (retain top 50%)
            let survivors: Vec<Board> = fitness
                .iter()
                .take(self.population_size / 2)
                .map(|(board, _)| board.clone())
                .collect();

            // Generate new population
            let mut new_population = Vec::new();

            // Crossover
            while new_population.len() < self.population_size {
                // Select two parents
                let parent1 = &survivors[rng.gen_range(0..survivors.len())];
                let parent2 = &survivors[rng.gen_range(0..survivors.len())];

                // Create offspring
                let child = self.crossover(parent1, parent2);
                new_population.push(child);
            }

            // Mutation
            for individual in new_population.iter_mut() {
                self.mutate(individual);
            }

            population = new_population;
        }

        // Return the best solution
        let (best_board, _) = population
            .iter()
            .map(|b| (b.clone(), b.calculate_score()))
            .max_by_key(|&(_, score)| score)
            .unwrap();

        (best_board, start_time.elapsed())
    }

    fn crossover(&self, parent1: &Board, parent2: &Board) -> Board {
        let mut rng = rand::thread_rng();
        let mut child = parent1.clone();

        // Combine regions randomly
        for region in parent1.regions.iter() {
            if rng.gen_bool(0.5) {
                // Take stars from parent2 for this region
                for &(row, col) in region {
                    if parent2.stars.contains(&(row, col)) {
                        child.stars.insert((row, col));
                    } else {
                        child.stars.remove(&(row, col));
                    }
                }
            }
        }

        child
    }

    fn mutate(&self, board: &mut Board) {
        let mut rng = rand::thread_rng();

        if rng.gen_bool(self.mutation_rate) {
            // Add a random mutation
            let row = rng.gen_range(0..board.size);
            let col = rng.gen_range(0..board.size);

            if board.stars.contains(&(row, col)) {
                board.stars.remove(&(row, col));
            } else {
                board.stars.insert((row, col));
            }
        }
    }
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;

    fn simple_table() -> Board {
        let size = 4;
        let mut regions = Vec::new();
        let region1: HashSet<(usize, usize)> = vec![(0,0), (0,1), (0,2), (0,3), (1,3)].into_iter().collect();
        let region2: HashSet<(usize, usize)> = vec![(1,0), (1,1), (1,2), (2,0)].into_iter().collect();
        let region3: HashSet<(usize, usize)> = vec![(3,0), (3,1), (3,2), (2,1), (2,2)].into_iter().collect();
        let region4: HashSet<(usize, usize)> = vec![(2,3), (3,3)].into_iter().collect();
        regions.push(region1);
        regions.push(region2);
        regions.push(region3);
        regions.push(region4);
        Board::new(size, regions, 1)
    }

    fn complex_table() -> Board {
        let size = 5;
        let mut regions = Vec::new();
        
        // create regions for testing
        let region1: HashSet<(usize, usize)> = vec![(0,0), (0,1), (0,2), (0,3), (0,4),
                                                (1,0), (1,1), (1,2), (1,3),
                                                (2,0), (2,1)
                                            ].into_iter().collect();
        let region2: HashSet<(usize, usize)> = vec![(1,4), (2,4)].into_iter().collect();
        let region3: HashSet<(usize, usize)> = vec![(2,2), (2,3), (3,2), (4,2)].into_iter().collect();
        let region4: HashSet<(usize, usize)> = vec![(3,0), (3,1), (4,0), (4,1)].into_iter().collect();
        let region5: HashSet<(usize, usize)> = vec![(3,3), (3,4), (4,3), (4,4)].into_iter().collect();

        regions.push(region1);
        regions.push(region2);
        regions.push(region3);
        regions.push(region4);
        regions.push(region5);

        Board::new(size, regions, 1)
    }

    #[test]
    fn test_pso() {
        let pso_solver = PSOSolver::new(50, 10000, 0.7, 1.2, 2.0);
        let mut board = simple_table();
        board.init_full_stars();
        let (pso_solution, pso_time) = pso_solver.solve(&board);
        pso_solution.print_board();
        println!("{} -- {:?} -- {}", pso_solution.calculate_score(), pso_time, pso_solution.is_valid())
    }

    #[test]
    fn test_evolutionary() {
        let mut board = complex_table();
        board.init_full_stars();

        let evo_solver = EvolutionarySolver::new(100, 200, 0.2);
        let (evo_solution, evo_time) = evo_solver.solve(&board);
        evo_solution.print_board();
        println!("{} -- {:?} -- {}", evo_solution.calculate_score(), evo_time, evo_solution.is_valid())
    }

    #[test]
    fn test_annealing() {
        let mut board = complex_table();
        board.initialize_random();

        let sa_solver = SimulatedAnnealingSolver::new(800.0, 0.95, 120);
        let (sa_solution, sa_time) = sa_solver.solve(&board);
        sa_solution.print_board();
        println!("{} -- {:?} -- {}", sa_solution.calculate_score(), sa_time, sa_solution.is_valid())
    }

    #[test]
    fn test_hybrid() {
        let mut board = complex_table();
        board.initialize_random();

        let hybrid_solver = HybridSolver::new(20, 0.5, 0.7, 0.2, 800.0, 0.95, 120);
        let (hybrid_solution, hybrid_time) = hybrid_solver.solve(&board);
        hybrid_solution.print_board();
        println!("{} -- {:?} -- {}", hybrid_solution.calculate_score(), hybrid_time, hybrid_solution.is_valid())
    }

    #[test]
    fn test_aco() {
        let mut board = simple_table();
        board.init_full_stars();
        let aco_solver = AntColonySolver::new(
        200,     
        300,    
        0.1,  
        1.0,
        2.0
        );
        let (solution, duration) = aco_solver.solve(&board);
        solution.print_board();
        println!("{} -- {:?} -- {}", solution.calculate_score(), duration, solution.is_valid())
    }

    #[test]
    fn benchmark() {
        // Initialize all solvers
        let aco_solver = AntColonySolver::new(200, 200, 0.1, 1.0, 2.0);
        let sa_solver = SimulatedAnnealingSolver::new(800.0, 0.95, 120);
        let pso_solver = PSOSolver::new(50, 10000, 0.7, 1.2, 2.0);
        let hybrid_solver = HybridSolver::new(20, 0.5, 0.7, 0.2, 800.0, 0.95, 120);
        let evo_solver = EvolutionarySolver::new(100, 200, 0.1);

    // Test configurations
        let test_cases = vec![
            ("Simple Table", simple_table()),
            ("Complex Table", complex_table()),
        ];
    
        // Number of runs per solver to get average performance
        let runs_per_solver = 3;
    
    // Store results for each solver
        #[derive(Default, Clone, Copy)]
        struct SolverStats {
            valid_solutions: usize,
            total_time: Duration,
            best_score: i32,
            avg_score: f64,
        }
    
        // Create results table
        let mut results: HashMap<(&str, &str), SolverStats> = HashMap::new();
    
        // Run benchmarks
        for (table_name, initial_board) in test_cases.iter() {
            println!("\nBenchmarking {}", table_name);
            println!("===================");

            // Test each solver
            for (solver_name, solver) in vec![
                ("ACO", Box::new(|board: &Board| aco_solver.solve(board)) as Box<dyn Fn(&Board) -> (Board, Duration)>),
                ("SA", Box::new(|board: &Board| sa_solver.solve(board))),
                ("PSO", Box::new(|board: &Board| pso_solver.solve(board))),
                ("Hybrid", Box::new(|board: &Board| hybrid_solver.solve(board))),
                ("Evolutionary", Box::new(|board: &Board| evo_solver.solve(board))),
            ] {
                println!("\nTesting {} solver", solver_name);
                let mut stats = SolverStats::default();
                stats.best_score = -999999;
                let mut total_score = 0;
                
                for run in 0..runs_per_solver {
                    println!("Run {}/{}", run + 1, runs_per_solver);

                    // Create a fresh board for each run
                    let mut board = initial_board.clone();
                    board.initialize_random();
                
                    // Solve
                    let (solution, duration) = solver(&board);
                    stats.total_time += duration;
                
                    // Update statistics
                    let score = solution.calculate_score();
                    total_score += score;
                    stats.best_score = stats.best_score.max(solution.calculate_score());
                
                    if solution.is_valid() {
                        stats.valid_solutions += 1;
                    }
                }
            
                // Calculate average score
                stats.avg_score = total_score as f64 / runs_per_solver as f64;
            
                // Store results
                results.insert((table_name, solver_name), stats.clone());
            
                // Print interim results
                println!("Results for {} on {}:", solver_name, table_name);
                println!("  Valid solutions: {}/{}", stats.valid_solutions, runs_per_solver);
                println!("  Average time: {:?}", stats.total_time / runs_per_solver as u32);
                println!("  Best score: {}", stats.best_score);
                println!("  Average score: {:.2}", stats.avg_score.clone());
            }
        }
    
        // Print final comparison table
        println!("\nFinal Results");
        println!("=============");
        println!("{:<15} {:<10} {:<15} {:<12} {:<12} {:<15}", 
            "Table", "Solver", "Avg Time (ms)", "Valid Sols", "Best Score", "Avg Score");
        println!("{}", "-".repeat(80));
    
        for table_name in ["Simple Table", "Complex Table"] {
            for solver_name in ["ACO", "SA", "PSO", "Hybrid", "Evolutionary"] {
                if let Some(stats) = results.get(&(table_name, solver_name)) {
                    println!("{:<15} {:<10} {:<15.2} {:<12} {:<12} {:<15.2}",
                        table_name,
                        solver_name,
                        stats.total_time.as_millis() as f64 / runs_per_solver as f64,
                        stats.valid_solutions,
                        stats.best_score,
                        stats.avg_score);
                }
            }
            println!("{}", "-".repeat(80));
        }
    }
}



fn main() -> Result<(), Error> {
    Ok(())
}
