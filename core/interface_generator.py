import os
from itertools import product
import numpy as np
import math
from pymatgen.io.cif import CifWriter
from ase.build import general_surface
from ase.spacegroup import crystal
from ase.visualize import view
from ase.lattice.surface import *
from ase.io import *
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
import argparse
import pymatgen as mg
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import Slab, SlabGenerator, ReconstructionGenerator
from pymatgen.analysis.substrate_analyzer import SubstrateAnalyzer, ZSLGenerator
from pymatgen.symmetry.analyzer import *
from core.utils.utils import *


def Struc_flip(struc):
    """
        Corrects the orientation of structures
    """
    struc_coords = struc.frac_coords
    z_frac_coords = struc_coords[:, 2]
    new_z_frac_coords = []
    for i in z_frac_coords:
        new_z_frac_coords.append(1 - i)

    struc_coords[:, 2] = new_z_frac_coords
    new_struc = Structure(struc.lattice.matrix, species=struc.species, coords=struc_coords)
    new_struc_space = SpacegroupAnalyzer(new_struc)
    new_struc_pri = new_struc_space.get_primitive_standard_structure()
    

    return new_struc_pri

def z_coords_selector(coords, layer_index):
    """
        Select the coordinate indices of a certain layer
    """
    
    z_coords = coords[:, [2]]
    new_z_coords = []
    for i in z_coords:
        if round(i[0] , 8)  not in new_z_coords:
            new_z_coords.append(round(i[0] , 8))
    sorted_z_coords = np.array(sorted(new_z_coords , reverse= False))
    selected_indices = []
    counter = 0
    for i in coords:
        if (round(i[2], 8) == round(sorted_z_coords[layer_index], 8)) or (
                round(i[2], 8) == round(sorted_z_coords[layer_index - 1], 8)) or (
                round(i[2], 8) == round(sorted_z_coords[layer_index + 1], 8)):
            selected_indices.append(counter)
        counter += 1

    return selected_indices

def coords_separator(coords, num, is_sub=True):
    """
        Separates a range of layes from top/bottom of structure
    """
    
    new_coords = []
    for i in coords:
        if round(i[0] , 8)  not in new_coords:
            new_coords.append([round(i[0] , 8)])
    coords = np.array(sorted(new_coords , reverse= is_sub))
    return coords[range(num)]

def dotproduct(v1, v2):
   return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def triangel_area(r1, r2, r3):
    """
        calculate the triangle area formed by three coordinates
    """

    a = np.linalg.norm(r2 - r1)
    b = np.linalg.norm(r3 - r1)
    c = np.linalg.norm(r3 - r2)
    s = (a + b + c) / 2
    # calculate the area
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    return area

def In_triangel(p, r1, r2,r3):
    """
        Checks if a selected point located in a certain atoms triangle
    """
    
    tot_area = triangel_area(r1,r2,r3)
    area1 = triangel_area(p, r1, r2)
    area2 = triangel_area(p, r1, r3)
    area3 = triangel_area(p, r2, r3)

    if round(tot_area, 3) == round(area1 + area2 + area3 , 3):
        return True
    else:
        return False

def fast_norm(a):
    """
    Much faster variant of numpy linalg norm
    """
    return np.sqrt(np.dot(a, a))
def vec_area(a, b):
    """
    Area of lattice plane defined by two vectors
    """
    return fast_norm(np.cross(a, b))


def reduce_vectors(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill
    """
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)

    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)

    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))

    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))

    return [a, b]

def dic_key_generator(array1, array2):
    list1 = []
    list2 = []
    for ar1 in array1:
        list1.append(np.sum(ar1))
    for ar2 in array2:
        list2.append(np.sum(ar2))
    mat1 = np.matrix(list1)
    mat2 = np.matrix(list2)
    mean1 = np.round(np.mean(mat1), 6)
    mean2 = np.round(np.mean(mat2), 6)
    tot_mean = np.round(mean1 + mean2 , 6)

    return tot_mean

def rel_strain(vec1, vec2):
    """
    Calculate relative strain between two vectors
    """
    return fast_norm(vec2) / fast_norm(vec1) - 1

def rel_angle(vec_set1, vec_set2):
    """
    Calculate the relative angle between two vector sets

    Args:
        vec_set1(array[array]): an array of two vectors
        vec_set2(array[array]): second array of two vectors
    """
    return vec_angle(vec_set2[0], vec_set2[1]) / vec_angle(
        vec_set1[0], vec_set1[1]) - 1



def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def vectors_misfit(vec_set1, vec_set2):
    """
    Calculate angle between two vectors
    """
    
    length_mf1 = np.absolute(rel_strain(vec_set1[0], vec_set2[0]))
    length_mf2 = np.absolute(rel_strain(vec_set1[1], vec_set2[1]))
    angle_mf = np.absolute(rel_angle(vec_set1, vec_set2))

    return [length_mf1, length_mf2, angle_mf]

def get_equiv_transformations(self, transformation_sets, film_vectors,
                                  substrate_vectors):

    """
        Monkey-patching the original function of pymatgen to generate the transformation matrices
    """

    calc_film_area = vec_area(film_vectors[0] , film_vectors[1])
    calc_sub_area = vec_area(substrate_vectors[0] , substrate_vectors[1])
    text_file2 = open(self.working_dir + "Ogre_output", "w")
    text_file = open(self.working_dir + "matrices_sets", "w")
    film_sub_sets = []
    # print("****************************************")
    for (film_transformations, substrate_transformations) in \
            transformation_sets:

        calc_i = film_transformations[0][0,0] * film_transformations[0][1,1]
        calc_j = substrate_transformations[0][0,0] * substrate_transformations[0][1,1]
        area_misfit = np.absolute(calc_film_area / calc_sub_area - float(calc_j) / calc_i)

        # Apply transformations and reduce using Zur reduce methodology
        films = [reduce_vectors(*np.dot(f, film_vectors))
                 for f in film_transformations]
        new_films = []
        for i in films:
            new_films.append(mat_clean(i))

        substrates = [reduce_vectors(*np.dot(s, substrate_vectors))
                      for s in substrate_transformations]

        new_substrates = []
        for i in substrates:
            new_substrates.append(mat_clean(i))
        # Check if equivelant super lattices

        for f, s in product(films, substrates):
            if self.is_newe_vectors(f, s):
                misfit_list = vectors_misfit(f, s)
                f_index = new_films.index(mat_clean(f))
                s_index = new_substrates.index(mat_clean(s))
                #"  ", area_misfit_list[print_num],
                print([film_transformations[f_index].tolist(), substrate_transformations[s_index].tolist()],
                      file=text_file)
                print("Area_misft:" ,np.round(area_misfit , 5),
                      "   Length misfit1:", np.round(misfit_list[0], 5),
                      "   Length misfit2:",np.round(misfit_list[1], 5) ,
                      "   Angle misfit:",np.round(misfit_list[2], 5) , file= text_file2)

                film_sub_sets.append(
                    [film_transformations[f_index].tolist(), substrate_transformations[s_index].tolist()])
                yield [f, s]



    text_file.close()
    text_file2.close()
    # os.remove(self.working_dir + "area_misfits")


def Interface_generator(Ini_sub_slab, Ini_film_slab, sub_tr_mat, film_tr_mat, distance, fparam  ):

    raw_ini_sub_slab_mat = np.array(Ini_sub_slab.lattice.matrix)
    raw_ini_film_slab_mat = np.array(Ini_film_slab.lattice.matrix)
    sub_reduction = reduce_vectors(raw_ini_sub_slab_mat[0], raw_ini_sub_slab_mat[1])
    film_reduction = reduce_vectors(raw_ini_film_slab_mat[0], raw_ini_film_slab_mat[1])
    reduced_sub_mat = np.array([sub_reduction[0], sub_reduction[1], raw_ini_sub_slab_mat[2]])
    reduced_film_mat = np.array([film_reduction[0], film_reduction[1], raw_ini_film_slab_mat[2]])
    red_Ini_sub_slab = Structure(mg.Lattice(reduced_sub_mat), Ini_sub_slab.species, Ini_sub_slab.cart_coords,
                                 to_unit_cell= True, coords_are_cartesian=True)
    red_Ini_film_slab = Structure(mg.Lattice(reduced_film_mat), Ini_film_slab.species, Ini_film_slab.cart_coords,
                                  to_unit_cell= True, coords_are_cartesian=True)

    red_Ini_sub_slab.make_supercell(scaling_matrix= scale_mat(sub_tr_mat) )
    red_Ini_film_slab.make_supercell(scaling_matrix= scale_mat(film_tr_mat))
    Ini_sub_mat = red_Ini_sub_slab.lattice.matrix
    Ini_film_mat = red_Ini_film_slab.lattice.matrix
    sub_r_vecs = reduce_vectors(Ini_sub_mat[0], Ini_sub_mat[1])
    film_r_vecs = reduce_vectors(Ini_film_mat[0], Ini_film_mat[1])
    sub_mat = np.array([sub_r_vecs[0], sub_r_vecs[1], Ini_sub_mat[2]])
    film_mat = np.array([film_r_vecs[0], film_r_vecs[1], Ini_film_mat[2]])

    AB_C = np.dot(np.cross(sub_mat[0], sub_mat[1]), sub_mat[2])
    if AB_C > 0:
        Is_right_handed = True
    else:
        #Is_right_handed = False
        Is_right_handed = True

    modif_sub_struc = mg.Structure(mg.Lattice(sub_mat), red_Ini_sub_slab.species, red_Ini_sub_slab.cart_coords,
                                   to_unit_cell= True, coords_are_cartesian=True)
    modif_film_struc = mg.Structure(mg.Lattice(film_mat), red_Ini_film_slab.species, red_Ini_film_slab.cart_coords,
                                    to_unit_cell= True, coords_are_cartesian=True)
    sub_sl_vecs = [modif_sub_struc.lattice.matrix[0], modif_sub_struc.lattice.matrix[1]]
    film_sl_vecs = [modif_film_struc.lattice.matrix[0], modif_film_struc.lattice.matrix[1]]
    film_angel = angle(film_sl_vecs[0], film_sl_vecs[1])
    sub_angel = angle(sub_sl_vecs[0], sub_sl_vecs[1])
    u_size = fparam * (np.linalg.norm(sub_sl_vecs[0])) + (1 - fparam) * (np.linalg.norm(film_sl_vecs[0]))
    v_size = fparam * (np.linalg.norm(sub_sl_vecs[1])) + (1 - fparam) * (np.linalg.norm(film_sl_vecs[1]))
    mean_angle = fparam * sub_angel + (1 - fparam) * film_angel
    sub_rot_mat = [[u_size, 0, 0], [v_size * math.cos(mean_angle), v_size * math.sin(mean_angle), 0],
                   [0, 0, np.linalg.norm(modif_sub_struc.lattice.matrix[2])]]
    film_rot_mat = [[u_size, 0, 0], [v_size * math.cos(mean_angle), v_size * math.sin(mean_angle), 0],
                    [0, 0, -np.linalg.norm(modif_film_struc.lattice.matrix[2])]]
    film_normal = np.cross(film_sl_vecs[0], film_sl_vecs[1])
    sub_normal = np.cross(sub_sl_vecs[0], sub_sl_vecs[1])
    film_un = film_normal / np.linalg.norm(film_normal)
    sub_un = sub_normal / np.linalg.norm(sub_normal)
    film_sl_vecs.append(film_un)
    L1_mat = np.transpose(film_sl_vecs)
    L1_res = [[u_size, v_size * math.cos(mean_angle), 0], [0, v_size * math.sin(mean_angle), 0], [0, 0, 1]]
    L1_mat_inv = np.linalg.inv(L1_mat)
    L1 = np.matmul(L1_res, L1_mat_inv)
    sub_sl_vecs.append(sub_un)
    L2_mat = np.transpose(sub_sl_vecs)
    L2_res = [[u_size, v_size * math.cos(mean_angle), 0], [0, v_size * math.sin(mean_angle), 0], [0, 0, -1]]
    L2_mat_inv = np.linalg.inv(L2_mat)
    L2 = np.matmul(L2_res, L2_mat_inv)
    sub_rot_lattice = mg.Lattice(sub_rot_mat)
    film_rot_lattice = mg.Lattice(film_rot_mat)
    r_sub_coords = np.array(modif_sub_struc.cart_coords)
    r_film_coords = np.array(modif_film_struc.cart_coords)

    for ii in range(len(r_sub_coords)):
        r_sub_coords[ii] = np.matmul(L2, r_sub_coords[ii])
    for ii in range(len(r_film_coords)):
        r_film_coords[ii] = np.matmul(L1, r_film_coords[ii])

    sub_slab = mg.Structure(sub_rot_lattice, modif_sub_struc.species, r_sub_coords, to_unit_cell= True,
                            coords_are_cartesian=True)
    film_slab = mg.Structure(film_rot_lattice, modif_film_struc.species, r_film_coords,to_unit_cell= True,
                             coords_are_cartesian=True)
    sub_sp_num = len(sub_slab.types_of_specie)
    film_sp_num = len(film_slab.types_of_specie)


    sub_slab_mat = np.array(sub_slab.lattice.matrix)
    film_slab_mat = np.array(film_slab.lattice.matrix)
    sub_slab_coords = sub_slab.cart_coords
    film_slab_coords = film_slab.cart_coords

    sub_slab_zmat = sub_slab_coords[:, [2]]
    film_slab_zmat = film_slab_coords[:, [2]]
    sub_slab_zmat = sub_slab_zmat - min(sub_slab_zmat)
    film_slab_zmat = film_slab_zmat - min(film_slab_zmat)
    sub_max_z = max(sub_slab_zmat)
    sub_min_z = min(sub_slab_zmat)
    modif_film_slab_zmat = film_slab_zmat + sub_max_z - sub_min_z  + distance
    film_slab_coords[:, [2]] = modif_film_slab_zmat
    sub_slab_coords[:,[2]] = sub_slab_zmat

    sub_max_z =  max(sub_slab_zmat)
    film_min_z = min(modif_film_slab_zmat)
    sub_max_list = coords_sperator_2(sub_slab_zmat, sub_sp_num , True)
    film_min_list = coords_sperator(modif_film_slab_zmat, film_sp_num , False)

    interface_coords = np.concatenate((sub_slab_coords, film_slab_coords), axis=0)
    interface_species = sub_slab.species + film_slab.species
    interface_latt = sub_slab_mat
    interface_latt[2][2] = abs(sub_slab_mat[2][2])  +  abs(film_slab_mat[2][2])  + distance

    Adding_val = 0.5 * (interface_latt[2][2] - max(interface_coords[:,[2]]))
    sub_max_list += Adding_val
    film_min_list += Adding_val
    sub_max_z +=  Adding_val
    film_min_z +=  Adding_val

    interface_coords[:,[2]] += 0.5 * (interface_latt[2][2] - max(interface_coords[:,[2]]) )
    sub_slab_coords[:, [2]] += Adding_val
    film_slab_coords[:, [2]] += Adding_val
    interface_lattice = mg.Lattice(interface_latt)
    interface_struc = mg.Structure(interface_lattice, interface_species, interface_coords,to_unit_cell= True,
                                   coords_are_cartesian=True)
    interface_struc = interface_struc.get_reduced_structure()

    return [interface_struc, [sub_sp_num , film_sp_num], sub_slab_coords, film_slab_coords, sub_slab , film_slab,
            Is_right_handed]

