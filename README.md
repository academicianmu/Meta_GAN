- 👋 Hi, I’m @academicianmu
- 👀 I’m interested in computer
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

# GAN Model Collection: Industrial Data Generation and Analysis Toolkit

## Project Introduction
This project contains multiple implementations of Generative Adversarial Network (GAN) models, focusing on the generation and analysis of industrial data (such as cement industry, paper industry). Through different GAN structures, it can generate industry data with real distribution characteristics, supporting application scenarios like data augmentation and privacy protection.

## Model List

| Model File | Description | Core Features |
|---------|------|---------|
| `gan.py` | Basic GAN model | 4-dimensional feature generation, simple network structure, suitable for introductory reference |
| `gan_with_meta.py` | GAN with meta-learning | Introduces meta-learning mechanism to support rapid adaptation to new data distributions |
| `gan_with_meta_nofilter.py` | Meta-learning GAN without data filtering | Removes data filtering steps to retain original data distribution |
| `gan_with_meta_only.py` | Single-feature meta-learning GAN | Focuses on single-feature (total emissions) generation |
| `gan_with_mc.py` | GAN with memory pool | Introduces sample memory pool to stabilize training process |
| `gan_with_mc_only.py` | Single-feature memory pool GAN | Single-feature generation + memory pool mechanism |
| `graph_gan.py` | Graph structure GAN | Generates both data features and feature correlation graphs (adjacency matrices) simultaneously |

## Core Features

1. **Data Processing**: Supports Excel data reading, cleaning, normalization, and handles missing values and outliers
2. **Model Training**: Various GAN training strategies, including basic training, meta-learning training, etc.
3. **Data Generation**: Generates industry data that conforms to real distribution
4. **Evaluation Mechanism**: Includes generated data evaluation based on mean and standard deviation

## Environmental Dependencies

- Python 3.7+
- PyTorch 1.7+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Key Functional Modules

### 1. Data Visualization
- Compares distribution of real and generated data through histogram
- Plots training loss curves of generator and discriminator
- Visualizes feature correlation through adjacency matrix heatmap (for `graph_gan.py`)

### 2. Model Architectures
- **Generator**: Multi-layer neural network with Sigmoid output to ensure values in [0,1] range
- **Discriminator**: Uses LeakyReLU activation function and Dropout for regularization
- **Graph Generator**: Specialized generator for adjacency matrix generation (in `graph_gan.py`)

### 3. Training Mechanisms
- Basic GAN training with BCE loss
- Meta-learning training with inner and outer loop optimization
- Joint training for data and graph generation (in `graph_gan.py`)

### 4. Evaluation Tools
- Monte Carlo evaluator to assess sample quality
- Statistical comparison (mean and standard deviation) between real and generated data
- Automatic saving of generated data and statistical information to CSV files

## Output Files
- Generated data files (e.g., `cement_data2.csv`, `graph_generated_data.csv`, `mc_generated_data.csv`)
- Visualization plots (data distribution comparison, loss curves, adjacency matrix heatmaps)

## Notes
- All generated data is forced to be non-negative to conform to industrial data characteristics
- The models focus on four key industrial features: total emissions, electricity consumption, bituminous coal consumption, and diesel consumption
- Different models can be selected based on specific needs for single/multi-feature generation or correlation analysis
