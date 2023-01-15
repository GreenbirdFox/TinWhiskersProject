# Tin_Whiskers_Project

## About the Project
To predict the probability of tin whiskers physically bridging between different exposed conductors on a printed wiring board (PWB), a user-friendly graphical user interface (GUI) based on Monte Carlo simulation was designed and developed using Python. This tool performs Monte Carlo simulations to randomly toss a user-defined number of detached metal whiskers on a “virtual” PWB, in which all exposed electrical conductors’ coordinates could be captured through image processing algorithms, and compute the frequency of detached whiskers causing a bridge, the frequency of conductor pairs being bridged, and the frequency of each conductor being bridged. Assisted with this information, the mission assurance team and reliability engineers are able to know how likely their fielded systems might be shorted by metal whiskers and which conductor pairs might have the highest probability of being short, and then make informed risk assessments.

## Monte Carlo Simulation Flowchart
![image](https://user-images.githubusercontent.com/119905396/212526312-aa73e83d-1908-478e-bf01-35d890e77a33.png)

## Graphical User Interface
### Input Section
![image](https://user-images.githubusercontent.com/119905396/212526327-5994852f-eda0-4f3d-9361-6ab2a31587a0.png)

### Output Section
![image](https://user-images.githubusercontent.com/119905396/212526342-4021b6af-0dd6-4d78-9a29-1ac46e808e36.png)
