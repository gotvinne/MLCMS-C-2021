
import pandas as pd
import numpy as np

raw_data = pd.read_csv("speeds.csv",delimiter=" ") # Reads file
organized_data = raw_data.groupby("pedestrianId") # Arrange data in terms of pedestrianId

#for key, value in organized_data:
#    print(key)

organized_data["pedestrianId"]

ped_speeds = np.empty(0)
for _,pedestrian in organized_data:
    ped_speeds = np.append(ped_speeds,np.mean(pedestrian["speedInAreaUsingAgentVelocity-PID6"]))

print(ped_speeds)