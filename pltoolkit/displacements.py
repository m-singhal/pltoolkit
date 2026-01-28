import numpy as np
import os

class GenerateDisplacements:

    def __init__(self, gs_poscar_path, normal_modes, atomic_masses, selected_indices, disp_list):
        
        self.gs_poscar_path = gs_poscar_path
        self.disp_list = disp_list
        self.normal_modes = normal_modes
        self.atomic_masses = atomic_masses
        self.selected_indices = selected_indices
        
        self.read_poscar()
        self.mass_factor = 1/np.sqrt(self.atomic_masses)
        for index in range(len(self.selected_indices)):
            self.renormalized_mode = (self.normal_modes[self.selected_indices[index]].T*self.mass_factor).T
            for i, disp_amt in enumerate(self.disp_list):
                new_positions = self.generate_displacements(self.cartesian_atomic_positions, disp_amt)
                self.write_poscar(new_positions, index+1, i+1)

    def read_poscar(self):
        with open(self.gs_poscar_path, "r") as f:
            self.lines = f.readlines()
            self.comment = self.lines[0].strip()
            self.scaling_factor = float(self.lines[1].strip())
            self.lattice_vectors = np.loadtxt(self.lines[2:5]) * self.scaling_factor
            self.atomic_species = self.lines[5].strip().split()
            self.num_atoms = np.fromstring(self.lines[6], sep=' ', dtype=int)
            self.lattice_type = self.lines[7].strip()
            self.atomic_positions = np.loadtxt(self.lines[8:8 + sum(self.num_atoms)])
            
            if self.lattice_type.lower().startswith('d'):
                self.cartesian_atomic_positions = np.dot(self.atomic_positions, self.lattice_vectors)
            else:
                self.cartesian_atomic_positions = self.atomic_positions
    
    def generate_displacements(self, cartesian_atomic_positions, disp_amt):
        new_positions = cartesian_atomic_positions + disp_amt*self.renormalized_mode
        print(disp_amt*self.renormalized_mode.min())
        return new_positions

    def write_poscar(self, new_positions, index, disp_index):
        os.makedirs(f"disp/disp-poscar-{index}", exist_ok=True)
        disp_poscar_path = f"./disp/disp-poscar-{index}/POSCAR-{disp_index}"
        with open(disp_poscar_path, "w") as f:
            f.write(f"{self.comment}\n")
            f.write(f"{self.scaling_factor}\n")
            np.savetxt(f, self.lattice_vectors, fmt="%.16f")
            f.write(" ".join(self.atomic_species) + "\n")
            f.write(" ".join(map(str, self.num_atoms)) + "\n")
            f.write("Cartesian\n")
            np.savetxt(f, new_positions, fmt="%.16f")