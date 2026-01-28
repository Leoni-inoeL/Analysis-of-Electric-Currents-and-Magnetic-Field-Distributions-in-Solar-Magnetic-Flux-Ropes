# Analysis-of-Electric-Currents-and-Magnetic-Field-Distributions-in-Solar-Magnetic-Flux-Ropes
(Bachelor's thesis)

Instructions on how to run the code:


finding_center.py
1. Clone the repository:
   git clone https://github.com/Leoni-inoeL/Analysis-of-Electric-Currents-and-Magnetic-Field-Distributions-in-Solar-Magnetic-Flux-Ropes.git
2. Change the path to the folder you want the data to be downloaded to in the file data_loader.py (line 37: sunpy_data_dir = "C:\\Users\\user\\sunpy\\data").
3. Run finding_center.py


magnetic_field_antlysis_tool.py


1. Open ParaView and load the file with magnetic field data.
2. "Tools" -> "ParaView Python Editor" -> Paste this script there -> Set base_path (line 102) as the directory you want the files to be saved to -> "File" -> "Run". 
