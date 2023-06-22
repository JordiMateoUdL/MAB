""" Main file for the project. """
from mab.case_study.slot_machines import BernoulliSlotMachines


if __name__ == '__main__':

    print("#" * 80)
    print("Bernoulli Slot Machines - Case Study")
    slot_machines = BernoulliSlotMachines()
    slot_machines.run_simulation(10000)
    slot_machines.report_results()
    print("#" * 80)
