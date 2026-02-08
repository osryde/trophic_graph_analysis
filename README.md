# 🌿 Food Web Analysis & Extinction Cascade Simulator

A comprehensive analysis of trophic networks (food webs) using Network Theory and unsupervised Machine Learning, with an interactive Streamlit application for simulating extinction cascades.

## 📋 Project Overview

This project investigates the structural and functional vulnerability of food webs through a methodological framework that integrates:

- **Network Theory**: Analysis of topological properties (connectance, modularity, trophic levels)
- **Energetic Criteria**: Simulation of secondary extinctions based on energy thresholds
- **Unsupervised Machine Learning**: Clustering of food webs based on emergent structural properties
- **Interactive Visualization**: Real-time extinction cascade simulation via Streamlit

### Key Features

- Analysis of food web topology (species, links, connectance, modularity)
- Community detection using the Louvain algorithm
- Extinction cascade simulation with configurable energy thresholds
- R₅₀ robustness calculation (proportion of primary extinctions needed to cause 50% total extinctions)
- Keystone species identification
- UMAP-based clustering of multiple food webs
- Interactive network visualization with Plotly

## 🚀 Getting Started

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

## 📊 Dataset

The food web data used in this project comes from the **Web of Life** ecological networks database:

🔗 **Source**: [https://www.web-of-life.es/map.php?type=7](https://www.web-of-life.es/map.php?type=7)

The dataset includes 33 food webs from various biomes:
- **Marine ecosystems**: Caribbean, Gulf of Mexico, Coral Reefs (Virgin Islands, Marshall Islands, Madagascar, Hawaii, New Caledonia, Okinawa)
- **Freshwater rivers**: Parana River (Brazil), Cerrado River (Brazil), streams in Maine, North Carolina, New Zealand
- **Lakes**: Itaipu Reservoir, Pantanal Oxbow Lakes

Each CSV file represents an adjacency matrix where:
- **Rows** = Prey (resources)
- **Columns** = Predators (consumers)
- **Values** = Interaction weights (energy flow)

## 📚 Scientific Background

This project is inspired by the energetic robustness framework described in:

> **Bellingeri, M., et al.** *"Energetic criteria to predict biodiversity loss and keystone species in empirical agricultural food webs"*.

The key concept is the **energetic threshold (th)**: a species becomes extinct when its energy intake falls below a fraction `th` of its initial intake. The **R₅₀** index measures the proportion of primary extinctions required to cause 50% total extinctions in the network.

### Robustness Formula

$$R_\alpha = \frac{E}{S}$$

Where:
- $E$ = Number of primary extinctions
- $S$ = Total number of species
- $\alpha$ = Target fraction of total extinctions (e.g., 0.5 for R₅₀)

## 🔬 Methods

1. **Network Construction**: Build directed weighted graphs from adjacency matrices
2. **Topological Analysis**: Calculate network metrics (connectance, modularity, trophic levels)
3. **Community Detection**: Identify functional compartments using Louvain algorithm
4. **Extinction Simulation**: Model cascade effects with configurable energy thresholds
5. **Robustness Analysis**: Compare random vs. targeted removal strategies
6. **Clustering**: Group food webs by structural features using UMAP + K-Means

## 📈 Results

The analysis reveals that emergent network properties (modularity, trophic chain length, energetic robustness) are stronger structural drivers than taxonomic or environmental classification. The identified clusters represent "bioenergetic organization archetypes" - universal patterns of matter and energy flow that nature adopts to ensure system stability.

## 📜 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- [Web of Life](https://www.web-of-life.es/) for providing the ecological networks database
- All the researchers who collected and published the original food web data (see `dataset/references.csv`)
