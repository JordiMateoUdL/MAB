import copy

from examples.slot_machine import SlotMachine
from mab.domain.bandit_Old import Bandit
from mab.simulator.benchmark_report import BenchmarkReporter
from mab.simulator.simulation import SimulationController
from mab.simulator.simulation_report import SimulationReporter
from mab.solvers.solvers import EpsilonGreedy

if __name__ == "__main__":

    EXPLORATION_RATIO = 0.15
    slot_machines_config = [
        {"true_reward": 1.5},
        {"true_reward": 2.0},
        {"true_reward": 3.0}
    ]

    slot_machines = []
    for config in slot_machines_config:
        slot_machines.append(SlotMachine(**config))
        
    bandit = Bandit(arms=slot_machines)
    solver =  EpsilonGreedy(bandit=copy.deepcopy(bandit), epsilon=0.1)
    
    simulator = SimulationController()
    simulator.run_simulation(num_time_steps=1000, solver=solver)
    simulation_results = simulator.get_results()
    reporter = SimulationReporter(simulation_results)
    print(reporter.generate_report())
    
    
    solvers = [
         EpsilonGreedy(bandit=copy.deepcopy(bandit), epsilon=0.1),
         EpsilonGreedy(bandit=copy.deepcopy(bandit), epsilon=0.2),
         EpsilonGreedy(bandit=copy.deepcopy(bandit), epsilon=0.3),
         EpsilonGreedy(bandit=copy.deepcopy(bandit), epsilon=0.4),
         EpsilonGreedy(bandit=copy.deepcopy(bandit), epsilon=0.5)
     ]
    
    benchmark_reporter = BenchmarkReporter()
    for solver in solvers:
        simulator = SimulationController()
        simulator.run_simulation(num_time_steps=1000, solver=solver)
        
        simulation_results = copy.deepcopy(simulator.get_results())
        benchmark_reporter.add_simulation_results(simulation_results, str(solver))
    
    
    benchmark_reporter.plot_cumulative_reward()
    print(benchmark_reporter.generate_report())

    
  
    



    
    # I want to access individual metrics for each solver
    # I want to compare solvers using the metrics
    

    
    
    
    
    
    

    
    

    # strategies = [
    #     PureExploitation(arms=copy.deepcopy(slot_machines)),
    #     PureExploration(arms=copy.deepcopy(slot_machines)),
    #     EpsilonGreedy(arms=copy.deepcopy(slot_machines),
    #                   epsilon=EXPLORATION_RATIO),
    #     UCB1(arms=copy.deepcopy(slot_machines), c=1.5),
    #     ThomsonSampling(arms=copy.deepcopy(slot_machines), threshold=2.5)
    # ]

    # plot_strategies(bandit_class=SlotMachineBandit,
    #                strategies=strategies,
    #                plot_type='regret',
    #                ylabel='Cumulative Regret',
    #                num_rounds=1000)

    # for strategy in strategies:
    #     bandit = SlotMachineBandit(strategy=strategy)
    #     best_arm = bandit.run(num_rounds=1000)

    #     metrics = BanditMetrics(bandit)
    #     metrics.plot_cumulative_rewards()
    #     metrics.plot_histogram_arm_selections()
