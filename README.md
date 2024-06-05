# Markov Ride Operator Project

# Optimizing Automated Mobility on-Demand Operation with Markovian Model : A Case Study of the Tel Aviv Metropolis in 2040 

Autonomous Mobility on Demand (AMoD) services offers numerous benefits, such as lower operating costs, due to reduced fuel and insurance costs and have no driver (Howard and Dai, 2014; Fraedrich and Lenz, 2014), which makes it extremely attractive for future development, thus, attracted a lot of attention in the literature. The AmoD services are becoming a reality and in the near future, in metropolitan cities like Tel Aviv, it is expected that such a service will attract more than 10% of daily trips demand (Nahmias-Biran et al., 2023). However, only a few studies successfully tested and evaluated a full  AMoD service on a large and realistic network simulating real-world conditions. 

AMoD services fulfill four main tasks: dispatching, routing, ridesharing, and rebalancing. Dispatching assigns vehicles to customers based on availability, proximity, and battery level. Routing optimizes routes for profitability, while ridesharing serves multiple riders with one vehicle, reducing energy use but complicating trip planning with multiple route calculations (Zardini et al., 2021). The rebalancing task involves repositioning empty vehicles to optimize responsiveness and serve future demand (Dai et al., 2021). It is especially important because AMoD systems experience imbalance when some areas have more demand than others (Pavone et al., 2012).   

In this study we utilize a combined trio of simulation tools: (1) SimMobility demand prediction simulator, (2) Aimsun Next road network simulator, and (3) Aimsun Ride operator tool. The predicted demand for private vehicles and AmoD requests was done using SimMobility simulator for Tel Aviv futuristic metropolis in 2040, while this   demand is executed using the Aimsun Next simulator. Demand-supply feedback is taking place so that travel times in the network are being updated and feed the demand repeatedly until convergence. Simulation outputs countian 1.2M routes of private cars on the large-scale network with an emphasis on their energy consumption. To create an efficient service framework for AMOD operation, we adopt a mathematical model of a Markov decision process (MDP). MDP allows us to optimize tasks such as pickup, rebalancing and charging under demand and energy consumption constraints.  

The output of the MDP model is function  which suggests the optimal action of a single vehicle in the AmoD fleet needs to perform (charging, pick-up, rebalancing) at a certain point in time. Finally, we design and execute an operator using the Ride tool that simulates the vehicles in the urban environment of Tel Aviv metropolis, along with other road users, performing battery and charging monitoring and sends the AMoD’s to tasks according to the optimal policy proposed by the MDP model. We compared this operator to two policy scenarios: (1) Rebalancing to the highest demand area  after drop-offs , and (2) Self-decision rebalancing of the AMoD vehicle after drop-offs. 


## Table of Contents
1. [Markov  Model](#data-directory-structure)
2. [Operator](#data-directory-structure)
3. [Anlysis Scripts](#data-directory-structure)
4. [Folders Directory Structure](#data-directory-structure)
5. [License](#license)
