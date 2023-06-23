# MAB Simulator Package

MAB Simulator is a Python package that provides a framework for simulating and comparing multi-armed bandit algorithms.

@WARNING: This is current work, so that I will update it frequently.

* Notes:
    * [MultiArmed Bandit notes](docs/MAB.md)
* Case Study:
    * [Slot Machines](docs/SlotMachine.md)

## Installation

You can install the MAB Simulator package using [Poetry](https://python-poetry.org/), a dependency management and packaging tool for Python.

```bash
# Clone the repository
git clone https://github.com/JordiMateoUdL/MAB.git

# Navigate to the project directory
cd MAB

# Install the dependencies
poetry install
```
## Usage
To use the MAB Simulator package, you need to follow these steps:

1. Create instances of the bandit arms that represent your problem.
2. Create a Bandit object and pass the list of arms to it.
3. Create instances of different solvers, such as EpsilonGreedySolver, UCB1Solver, or ThomsonSamplingSolver, and pass the Bandit object to them. Also, you can extend the Solver class to compare other solvers or strategies.
4. Create a Simulator object and pass the Bandit object and the solvers list to it.
5. Run the simulation by calling the run_simulation() method of the Simulator object.
Retrieve the results using the get_results() method of the 
6. Simulator object.
7. Analyze and visualize the results using the available plotting functions.

See ```main.py```and ```case_study/slot_machine.py``` for a basic usage example.

## Contributing

We welcome contributions to the MAB Simulator package! If you would like to contribute, please follow these steps:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or bug fix:
3. Make your changes, ensuring that your code follows the project's coding conventions and standards.
4. Write tests to cover your changes and ensure that existing tests pass:
5. Commit your changes with a descriptive commit message.
6. Push your branch to your forked repository:
7. Open a Pull Request (PR) against the `main` branch of this repository. Provide a clear and detailed description of your changes in the PR, and reference any related issues or discussions.

Once your PR is submitted, it will be reviewed by the project maintainers. They may provide feedback or request further changes. Once the changes are approved, your code will be merged into the `main` branch.

### Code Style

Please adhere to the project's coding conventions and standards when making contributions. Ensure that your code follows consistent indentation, variable naming, and commenting practices. If possible, run code formatters or linters to automatically enforce these standards.





