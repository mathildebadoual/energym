# EnerGym

[![Build Status](https://travis-ci.org/mathildebadoual/energym.svg?branch=master)](https://travis-ci.org/mathildebadoual/energym)  [![Coverage Status](https://codecov.io/gh/mathildebadoual/energym/branch/master/graph/badge.svg)](https://codecov.io/gh/mathildebadoual/energym)
  
Power Systems environment for [OpenAI Gym](https://gym.openai.com/) developed at [eCAL](https://ecal.berkeley.edu/), UC Berkeley.


# Installation

```bash
cd energym
pip install -e .
```

# Running 

```bash
import gym
import energym

env = gym.make('battery-v0')
```

# Existing Environments

- "battery-v0": Environment of a simple battery 
- "energy_market-v0": Environment of a simple energy market with external data taken from CAISO.
- "energy_market_battery-v0": Environment of an energy market with a battery system (inputs are the bids of the battery owner in the market, the output is the new state of charge of the battery with the result of the clearance from the ISO)
- "spm_environment-v0": battery cell environment

For more information please visit the [Wiki](https://github.com/mathildebadoual/energym/wiki)

# Contact 
Mathilde Badoual: mathilde.badoual@berkeley.edu
Saehong Park: sspark@berkeley.edu
