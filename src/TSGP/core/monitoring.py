import time
from pathlib import Path
from deap import tools
import numpy as np

class Monitor:
    """Handles logging for both console output and file storage."""
    def __init__(self, config: dict, console_out: bool = True):
        self.config = config
        self.console_out = console_out
        self.start_time = time.time()
        
        # Set up statistics
        self.stats = self._register_stats()
        # Create separate logbooks
        self.monitor_logbook = tools.Logbook()
        self.printer_logbook = tools.Logbook()
        
        # Set up headers
        self._setup_headers()
        
        # Initialize counters
        self.generation = 0

    def _setup_headers(self):
        """Set up headers for both logbooks."""
        # Monitor logbook (detailed)
        self.monitor_logbook.header = [
            'seed', 'generation', 'best_fitness', 'best_fitness_test', 
            'fitness_avg', 'fitness_test_avg', 'fitness_25percentile', 
            'fitness_median', 'fitness_75percentile', 'size_avg', 'size_best', 
            'unique_rate', 'semantic_distance_25percentile', 
            'semantic_distance_median', 'semantic_distance_75percentile', 
            'time_used', 'false_expressions',
        ]
        
        # Printer logbook (clean console output)
        self.printer_logbook.header = [
            'generation', 'best_fitness','best_fitness_test', 'size_avg', 'size_best', 'semantic_distance_median', 'time_used',
        ]

        if self.config['algorithm'] == 'data_generator':
            self.printer_logbook.header.remove('semantic_distance_median')
            self.printer_logbook.header.append('Archive Size')
        
    def _register_stats(self):
        """ Register statistics to be tracked."""
        stats_fitness = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fitness_test = tools.Statistics(lambda ind: ind.fitness_test.values)

        stats_semantic_distance = tools.Statistics(lambda ind: getattr(ind, 'semantic_distance', np.nan))

        # Defensive height accessor: some malformed or non-standard individuals may
        # cause deap.gp.PrimitiveTree.height to raise (e.g. due to internal stack
        # state). Use a try/except fallback to a sensible size metric.
        def _safe_height(ind):
            try:
                return ind.height
            except Exception:
                # Fall back to length if available, else 0
                try:
                    return len(ind)
                except Exception:
                    return 0

        stats_depth = tools.Statistics(_safe_height)
        stats_size = tools.Statistics(len)
        stats = tools.MultiStatistics(fitness=stats_fitness,
                                        fitness_test=stats_fitness_test,
                                        semantic_distance=stats_semantic_distance,
                                        depth=stats_depth,
                                        size=stats_size)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("25percentile", np.percentile, q=25)
        stats.register("50percentile", np.median)
        stats.register("75percentile", np.percentile, q=75)
        stats.register("sum", np.sum)
        return stats

    def compile(self, engine):
        """ Compile and log statistics for the current generation."""
        current_time = time.time() - self.start_time
    
        record = self.stats.compile(engine.population)
        
        
        monitor_record = {
            'seed': self.config['random_seed'],
            'generation': self.generation,
            'best_fitness': engine.best_individual.fitness.values[0],
            'best_fitness_test': engine.best_individual.fitness_test.values[0],
            'fitness_avg': record['fitness']['avg'], 
            'fitness_test_avg': record['fitness_test']['avg'],  
            'fitness_25percentile': record['fitness']['25percentile'],
            'fitness_median': record['fitness']['50percentile'],
            'fitness_75percentile': record['fitness']['75percentile'],
            'size_avg': record['size']['avg'],  
            'size_best': len(engine.best_individual),
            'unique_rate': round(len({str(ind) for ind in engine.population}) / len(engine.population), 3),
            'semantic_distance_25percentile': record['semantic_distance']['25percentile'],
            'semantic_distance_median': record['semantic_distance']['50percentile'],
            'semantic_distance_75percentile': record['semantic_distance']['75percentile'],
            'time_used': current_time,
        }

        monitor_record['false_expressions'] = getattr(engine, 'false_expressions_counter', 0)

        printer_record = {
            'generation': self.generation,
            'best_fitness': engine.best_individual.fitness.values[0],
            'best_fitness_test': engine.best_individual.fitness_test.values[0],
            'size_avg': record['size']['avg'],
            'size_best': len(engine.best_individual),
            'semantic_distance_median': record['semantic_distance']['50percentile'],
            'time_used': current_time,
        }

        if engine.config['algorithm'] == 'data_generator':
            archive_size = len(engine.archive)
            max_size = engine.config['data_generator']['num_samples_archive']
            printer_record['Archive Size'] = f"{archive_size} / {max_size}"
        
        # Record to both logbooks
        self.monitor_logbook.record(**monitor_record)
        self.printer_logbook.record(**printer_record)
        
        # Print to console if requested
        if self.console_out:
            print(self.printer_logbook.stream)
        
        # Increment generation counter
        self.generation += 1

    def save_results(self, config):
        """Save monitor logbook to CSV file."""
        import pandas as pd
        
        folder_path = config['monitor_folder']
        random_seed = config['random_seed']
        algorithm = config['algorithm']
        dataset = config['problem']['dataset']
        default_path = Path(folder_path) / f"monitor_run_{algorithm}_{random_seed}_{dataset}.csv"
        
        # Determine run_id from config if available
        monitor_path = config.get('monitor_path', default_path)

        # Convert logbook to DataFrame
        records = [dict(record) for record in self.monitor_logbook]
        df = pd.DataFrame(records)
        df.to_csv(monitor_path, index=False)
        
        if self.console_out:
            print(f"Monitor data saved to: {monitor_path}")



    