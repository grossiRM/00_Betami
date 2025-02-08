# Bacteria attachment PET data analysis
Python codes and data for result and figure generation in paper " In situ measurements of dynamic bacteria transport and attachment in heterogeneous sand-packed columns". 

All Python code files (.py), and data files (.csv, .npy) should be downloaded and stored in the same folder (working directory) in order for run code files successfully.

## File description

Three-dimensional (3D) raw data from PET imaging of two column experiments (tracer and E. coli bacteria injections) were pre-processed by coarsening voxel dimensions, normalizing the concentration. The pre-processed tracer and bacteria raw data are stored in **"tracer_PET_data_processed.npy"** and **"bacteria_PET_data_processed.npy"**, respectively. 

The main code to analyze PET data and calculate bacteria attachment and attachment rate coefficients is **"pet_data_analysis.py"**. Functions required in this code are imported from **"pet_analysis_functions.py"** file and the tracer and bacteria pre-processed data above are input into this code file for analysis. This code is used for:
- Figure 2 plotting: Concentrations of tracer and bacteria from PET are produced at each time (t= 2,5,8 min since pule injection) to creat time-course concentration maps in the column.
- Figure 5, bottom graph plotting: The 3D map of attached bacteria concentration (S*/C<sub>0</sub>) is calculated and 2D center-slice average attachment is plotted. The 3D attachment data will be saved in a separate file for later use in the numerical model, and is also provided ready-to-use as **"bacteria_attachment_map_3d.csv"** file.
- Figures 3 and 4 plotting: This code file also calculates the 3D attachment rate coefficient (k<sub>f</sub>). Figure 3 is plotted based on the 2D slice-average k<sub>f</sub> and the distribution of k<sub>f</sub> values in the 3D column is plotted as a probability density function to generate Figure 4. The 3D attachment rate coefficient (k<sub>f</sub>) data will be saved in a separate file called **"kf_distribution_3d.csv"** also provided.

Next, the code to generate a 3D numerical model is in **"numerical_model.py"**. It is used to test the bacterial attachment during transport in a 3D column and compared with the experimental results to validate the kf calculation approach in **"pet_data_analysis.py"**. Functions required in this code are imported from **"kf_distribution_functions.py"**, **"first_order_attachment_3D_model_functions.py"**, and **"plotting_functions.py"**. Given similar column geometry, sediment porosity and dispersivities, and transport conditions, the numerical model predicts bacteria attachment (S*/C<sub>0</sub>) in 3D in two scenarios: 
1. a homogeneous column with a constant single k<sub>f</sub> averaged from all kf data in **"kf_distribution_3d.csv"** and 
2. a heterogeneous column parameterized with the attachment rate coefficient measured from PET by inputting the **"kf_distribution_3d.csv"** file.

Results of predicted S*/C<sub>0</sub> from the model is plotted in 2D as Figure 5, middle. Experimental S*/C<sub>0</sub> from **"bacteria_attachment_map_3d.csv"** file is loaded in this code to generate Figure 5 bottom plot for comparison with the numerical model result. From the experimental and numerical model results, the 1D average attachment profiles are also calculated and plotted in Figure 5, top plot.

# Supporting Information Data
The **"Supporting Information_Data.xlsx"** file includes results from radiolabeling batch experiments and radioactivity in effluent autosamples during PET experiments that are used to plot Figure S2 and S3 in the Supporting Information, respectively. Workbook "Fig S2_uptake and retention" contains a description of all radiolabeling batch experiments, followed by time-course measurements of the F18-FDG uptake and retention percentage by E. coli which were measured from triplicate samples. Uptake and retention results from these experiments are used to plot Figure S2 in the Supporting Information. Workbook "Fig S3_Effluent radioactivity" contains measurements of radioactivity in the bulk solution and bacteria from the effluent samples during bacteria injection in PET column experiment that are used to plot Figure S3 in the SI.


