import subprocess as sp
import os
import time 
import numpy as np

path_calculation = os.path.expanduser("/scratch/pawsey1141/uqmsin17/02rbn/anharmonicity")

num_folders = sp.run(
    'find . -maxdepth 1 -type d -name "disp-poscar-*" | wc -l',
    cwd=f"{path_calculation}/disp",
    shell=True,
    capture_output=True,
    text=True
)
num_folders = int(num_folders.stdout.strip())

num_disps = sp.run(
    "ls POSCAR-* | wc -l",
    cwd=f"{path_calculation}/disp/disp-poscar-1",
    shell=True,
    capture_output=True,
    text=True
)
num_disps = int(num_disps.stdout.strip())


total_num_poscars = num_folders * num_disps
print(f"Total number of POSCAR files to process: {total_num_poscars}")

for i in range(1, num_folders + 1):
    sp.run(
        f"for j in $(seq 1 {num_disps}); do mkdir -p $j; mv POSCAR-$j ./$j/POSCAR; ln -sf /scratch/pawsey1141/uqmsin17/02rbn/anharmonicity/inputs/* ./$j; done", cwd=f"{path_calculation}/disp/disp-poscar-{i}", shell=True
    )

if total_num_poscars <= 500:
    job_id = []
    for i in range(1, num_folders +1):
        for j in range(1, num_disps +1):
            submit = sp.run(
                f"sbatch run.sh",
                cwd=f"{path_calculation}/disp/disp-poscar-{i}/{j}",
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"Submitted disp-poscar-{i}/{j}")
            job_id.append(submit.stdout.strip().split()[-1])
            stat = sp.run(["squeue", "-u", "uqmsin17"], capture_output=True, text=True)
    while any(job in stat.stdout.split() for job in job_id):
        time.sleep(30)
        stat = sp.run(["squeue", "-u", "uqmsin17"], capture_output=True, text=True)
    print("\n")
    print("Jobs completed!")

else:
    num_folders_a_batch = (500 // num_disps) + 1
    a = num_folders / num_folders_a_batch
    if a > int(a):
        num_batches = int(a) + 1
    else:
        num_batches = int(a)
    
    schedule = []
    for b in range(num_batches):
        inner_schedule = []
        for c in range(b*num_folders_a_batch + 1, (b+1)*num_folders_a_batch + 1):
            if c<= num_folders:
                inner_schedule.append(c)
        schedule.append(inner_schedule)

    for batch in schedule:
        job_id = []
        for i in batch:
            for j in range(1, num_disps +1):
                submit = sp.run(
                    f"sbatch run.sh",
                    cwd=f"{path_calculation}/disp/disp-poscar-{i}/{j}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                print(f"Submitted disp-poscar-{i}/{j}")
                job_id.append(submit.stdout.strip().split()[-1])
                stat = sp.run(["squeue", "-u", "uqmsin17"], capture_output=True, text=True)
        while any(job in stat.stdout.split() for job in job_id):
            time.sleep(30)
            stat = sp.run(["squeue", "-u", "uqmsin17"], capture_output=True, text=True)
        print("\n")
        print("Batch completed!")
    print("Jobs completed!")

disp_energies = np.zeros((num_folders, num_disps))
for i in range(1, num_folders +1):
    for j in range(1, num_disps +1):
        with open(f"{path_calculation}/disp/disp-poscar-{i}/{j}/vasp.out", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "E0=" in line:
                    disp_energies[i-1, j-1] = float(line.strip().split()[4])
np.save(f"{path_calculation}/disp_energies.npy", disp_energies)

print("Energies saved to disp_energies.npy")