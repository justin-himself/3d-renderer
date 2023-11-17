import cProfile
import pstats
import csv
import io
from test.main import *


pr = cProfile.Profile()

origin_mesh = load_obj("test/teapot.obj")
print(origin_mesh.shape)


pr.enable()

time1 = time.time()
for i in range(10):
    rotation_animation(origin_mesh, frame, i)
time2 = time.time()

pr.disable()

print(time2-time1)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

data = s.getvalue().split("\n")

for idx, line in enumerate(data):
    if "ncalls" in line:
        data = data[idx+1:]
        break

profile_data = []

prefix_list = [
    "/opt",
    "/homebrew",
    "/Caskroom",
    "/miniforge",
    "/base",
    "/envs",
    "/3d_engine",
    "/lib",
    "/python3.9",
    "/site-packages",
    "/Users",
    "/justin",
    "/Library",
    "/CloudStorage",
    "/OneDrive-ualberta.ca",
    "/Courses",
    "/ENCMP 100",
    "/Contest",
    "/Code",
]

for line in data:
    segs = line.strip().split(" ")
    segs = [seg.strip() for seg in segs]
    segs = [segs for segs in segs if segs != ""]
    segs, func_name = segs[:5], " ".join(segs[5:])
    for prefix in prefix_list:
        func_name = func_name.replace(prefix, "")
    segs = segs[:5] + [func_name]
    if segs != ['']:
        profile_data.append(segs)


# Save data to data.csv
with open("test/profiler.csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["ncalls", "tottime", "percall",
                        "cumtime", "percall", "function"])
    csv_writer.writerows(profile_data)

print("Data saved to profiler.csv")
