import os
from core.interface_generator import *
from core.slab_generator import *
from core.overlap_OF import *
from core.passivator import *
from core.utils.config import load_config
from core.Miller_search import *
import argparse
from itertools import product
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--path', dest='path', type=str, default='./test/inorganic/')
    parser.add_argument('--isorganic', dest='isorganic',
                        type=bool, default=True)
    parser.add_argument('--repair', dest='repair', type=bool, default=False)
    return parser.parse_args()

'''
Usage: 
python main.py --path /test/organic/
python main.py --path /test/organic/ --repair True
python main.py --path /test/inorganic/
'''


def main():
    # ====================================================================================
    # -------------------------------- Load Config File ----------------------------------
    # ====================================================================================

    args = parse_arguments()
    working_dir = args.path
    config_file_path = os.path.join(working_dir, 'ogre_interfaces.ini')
    config_dict = load_config(config_file_path)

    sub_name = os.path.join(working_dir, config_dict['system']['sub_path'])
    film_name = os.path.join(working_dir, config_dict['system']['film_path'])
    file_format = config_dict['system']['format']

    specify_termination = config_dict['methods']['specify_termination']
    shift_gen = config_dict['methods']['surface_matching']
    search_algorithm = config_dict['methods']['search_algorithm']
    is_miller_search = config_dict['methods']['miller_index_scan']
    passivate = config_dict['methods']['passivate']

    is_organic = config_dict['parameters']['is_organic']
    sub_termination = config_dict['parameters']['sub_termination']
    film_termination = config_dict['parameters']['film_termination']
    passivate_sub = config_dict['parameters']['passivate_sub']
    passivate_film = config_dict['parameters']['passivate_film']
    sub_miller = config_dict['parameters']['sub_miller_index']
    film_miller = config_dict['parameters']['film_miller_index']
    max_sub_index, max_film_index = config_dict['parameters']['max_search_index']
    sub_layers = config_dict['parameters']['sub_layers'][0]
    film_layers = config_dict['parameters']['film_layers'][0]
    vacuum = config_dict['parameters']['vacuum_size']
    max_area = config_dict['parameters']['max_interface_area']
    max_area_ratio_tol, max_length_tol, max_angle_tol = config_dict['parameters']['tolarence']
    distance = config_dict['parameters']['interfacial_distance']
    max_number_atoms = config_dict['parameters']['max_number_atoms']
    structure_index = config_dict['parameters']['structure_index']
    x_shift = config_dict['parameters']['x_shift']
    y_shift = config_dict['parameters']['y_shift']
    z_shift = config_dict['parameters']['z_shift']

    if passivate_sub:
        sub_layers += 1

    if passivate_film:
        film_layers += 1

    fparam = 1
    
    # ===================================================================================================
    # ---------------------------------------------- Start ----------------------------------------------
    # ===================================================================================================

    ZSLGenerator.working_dir = working_dir
    ZSLGenerator.get_equiv_transformations = get_equiv_transformations_Sam_2
    ase_substrate = read(sub_name)
    ase_film = read(film_name)

    pmg_substrate = Structure.from_file(sub_name, primitive=False)
    pmg_film = Structure.from_file(film_name, primitive=False)

    ZSL = ZSLGenerator(max_area_ratio_tol=max_area_ratio_tol, max_area=max_area, max_length_tol=max_length_tol,
                       max_angle_tol=max_angle_tol)
    Sub_analyzer = SubstrateAnalyzer(ZSL, film_max_miller=10)

    if not is_miller_search:
        substrate_slabs = Possible_surfaces_generator(ase_substrate, sub_miller, sub_layers, vacuum, working_dir)
        film_slabs = Possible_surfaces_generator(ase_film, film_miller, film_layers, vacuum, working_dir)

        Match_finder = Sub_analyzer.calculate(film=pmg_film, substrate=pmg_substrate, substrate_millers=[sub_miller],
                                              film_millers=[film_miller])
        Match_list = list(Match_finder)

    if is_miller_search:

        sub_indices = get_unique_indicies(max_index=max_sub_index)
        film_indices = get_unique_indicies(max_index=max_film_index)
        miller_search(Sub_analyzer, pmg_substrate, pmg_film, sub_indices, film_indices, working_dir)

    sets = open(working_dir + "matrices_sets")
    film_sub_matices = []
    sets = sets.readlines()
    for x in sets:
        x = x.replace("[", "")
        x = x.replace("]", "")
        x = x.split(",")
        film_mat = [[float(x[0]), float(x[1])], [float(x[2]), float(x[3])]]
        sub_mat = [[float(x[4]), float(x[5])], [float(x[6]), float(x[7])]]
        film_sub_matices.append([film_mat, sub_mat])

    nStruc = max_number_atoms
    if len(film_sub_matices) <= nStruc:
        nStruc = len(film_sub_matices)

    if not is_miller_search:
        for i in range(len(substrate_slabs)):
            for j in range(len(film_slabs)):
                print("Generating interfaces with substrate termination {0} and film termination {1}".format(i, j))
                ext = "Interfaces_Sub_" + str(i) + "_Film_" + str(j)
                ext_Dir = os.path.join(working_dir, ext)
                if (os.path.isdir(ext_Dir) == False):
                    os.mkdir(ext_Dir)

                ini_sub_slab = substrate_slabs[i]
                ini_film_slab = film_slabs[j]

                k_num = -1
                k = 0
                delta_list, int_new_coords, interfaces_list = [], [], []
                struc_matcher = StructureMatcher(ltol=0.001, stol=0.001, angle_tol=0.001)

                first_iteration = True
                while (k_num < nStruc - 1) and (k < len(film_sub_matices)):

                    Interface = Interface_generator(substrate_slabs[i], film_slabs[j], film_sub_matices[k][1],
                                                    film_sub_matices[k][0], distance, fparam)
                    k += 1
                    Iface = Interface[0]
                    sub_coords = Interface[2]
                    film_coords = Interface[3]
                    sub_slab_struc = Interface[4].get_reduced_structure()
                    film_slab_struc = Interface[5].get_reduced_structure()

                    interface_red_struc = Iface.get_reduced_structure()
                    interface_primitive_struc = interface_red_struc.get_primitive_structure()

                    interface_coords = interface_primitive_struc.frac_coords
                    interface_species = interface_primitive_struc.species

                    ############################
                    # Slab passivation
                    ############################

                    if passivate:
                        sub_slab_pp = passivator(
                            sub_slab_primitive,
                            top=passivate_sub,
                            bot=passivate_sub,
                            symmetrize=True,
                        )
                        film_slab_pp = passivator(
                            film_slab_primitive,
                            top=passivate_film,
                            bot=passivate_film,
                            symmetrize=True,
                        )
                        interface_pp = passivator(
                            interface_primitive_struc,
                            top=passivate_film,
                            bot=passivate_sub,
                            symmetrize=True,
                        )
                    else:
                        sub_slab_pp = copy.deepcopy(sub_slab_primitive)
                        film_slab_pp = copy.deepcopy(film_slab_primitive)
                        interface_pp = copy.deepcopy(interface_primitive_struc)

                    ############################
                    # Structure duplicate removal
                    ############################
                    duplicate = False
                    if len(interfaces_list) != 0:
                        for pre_int in interfaces_list:
                            is_dup = struc_matcher.fit(pre_int, interface_pp)
                            if (is_dup == True):
                                duplicate = True
                                break

                    if duplicate:
                        continue
                    else:
                        interfaces_list.append(interface_pp)
                        k_num += 1
                        print(k_num, flip_in,film_sub_matices[k][1],film_sub_matices[k][0] )


                    if passivate_sub:
                        sub_layers -= 1

                    Poscar(interface_pp.get_primitive_structure()).write_file(
                        ext_Dir + "/POSCAR_interface_" + str(k_num) + "_" + str(sub_layers) + "_" +
                        str(film_layers), direct=True)

                    Poscar(sub_slab_pp).write_file(
                        ext_Dir + "/POSCAR_substrate_" + str(k_num) + "_" + str(sub_layers),
                        direct=False)

                    Poscar(film_slab_pp).write_file(
                        ext_Dir + "/POSCAR_film_" + str(k_num) + "_" + str(film_layers),
                        direct=False)


                    OL_sub_Input = PBC_coord_gen(sub_slab_struc, ext_Dir, k_num, 0,
                                                 1, 1, file_gen=True)
                    OL_film_Input = PBC_coord_gen(film_slab_struc, ext_Dir, k_num, 1,
                                                  1, 1, file_gen=True)
                    OL_interface_Input = PBC_coord_gen(interface_red_struc, ext_Dir, k_num, 2,
                                                       1, 1, file_gen=True)

                    sub_OL_vol = Mont_geo(OL_sub_Input[0], OL_sub_Input[1])
                    film_OL_vol = Mont_geo(OL_film_Input[0], OL_film_Input[1])
                    interface_OL_vol = Mont_geo(OL_interface_Input[0], OL_interface_Input[1])

                    sub_sp_vol = 0
                    film_sp_vol = 0
                    for sp in sub_slab_struc.species:
                        sub_sp_vol += sphe_vol(rad_dic[str(sp)])
                    for sp in film_slab_struc.species:
                        film_sp_vol += sphe_vol(rad_dic[str(sp)])

                    sub_OL = sub_OL_vol / sub_sp_vol
                    film_OL = film_OL_vol / film_sp_vol
                    average_OL = (sub_OL + film_OL) / 2

                    sub_coords = Interface[2]
                    film_coords = Interface[3]

                    if shift_gen:

                        if x_shift != -1:
                            x_grid_shift = np.linspace(x_shift[0], x_shift[1], x_shift[2])
                            x_ref = film_coords[:, 0].copy()
                        else:
                            x_grid_range = [0]
                        if y_shift != -1:
                            y_grid_shift = np.linspace(y_shift[0], y_shift[1], y_shift[2])
                            y_ref = film_coords[:, 1].copy()
                        else:
                            y_grid_range = [0]
                        if z_shift != -1:
                            z_grid_shift = np.linspace(z_shift[0], z_shift[1], z_shift[2])
                            z_ref = film_coords[:, 2].copy()
                        else:
                            z_grid_range = [0]

                        ext2 = "Interfaces_Sub_" + str(i) + "_Film_" + str(j) + "/Shifts_Iface_" + str(k_num)
                        ext_Dir2 = os.path.join(working_dir, ext2)
                        if (os.path.isdir(ext_Dir2) == False):
                            os.mkdir(ext_Dir2)

                        Score_array = np.zeros((len(x_grid_range), len(y_grid_range), len(z_grid_range)))
                        score_list = []
                        interface_latt = Iface.lattice.matrix

                        for x_ind in range(len(x_grid_range)):
                            for y_ind in range(len(y_grid_range)):
                                for z_ind in range(len(z_grid_range)):

                                    if x_shift != -1:
                                        film_coords[:, 0] = x_ref + x_grid_range[x_ind]
                                    if y_shift != -1:
                                        film_coords[:, 1] = y_ref + y_grid_range[y_ind]
                                    if z_shift != -1:
                                        film_coords[:, 2] = z_ref + z_grid_range[z_ind]

                                    int_coords = np.concatenate((sub_coords, film_coords), axis=0)
                                    new_int = Structure(Iface.lattice.matrix, Iface.species, int_coords,
                                                        coords_are_cartesian=True)
                                    reduced_int = new_int.get_reduced_structure()
                                    reduced_int_space = SpacegroupAnalyzer(reduced_int)
                                    primitive_int = reduced_int_space.get_primitive_standard_structure()
                                    Poscar(primitive_int).write_file(ext_Dir2 + "/POSCAR_Iface_" + str(k_num)
                                                                   + "_x_" + str(int(x_grid_range[x_ind] * 10)) + "_y_" + str(
                                        int(y_grid_range[y_ind] * 10)) + "_z_" + str(int(z_grid_range[z_ind] * 10)), direct=False)

                                    OL_Input = PBC_coord_gen(primitive_int, ext_Dir2, k, int(x_grid_range[ii] * 10),
                                                             int(y_grid_range[jj] * 10), int(z_grid_range[kk] * 10),
                                                             file_gen=True)
                                    vec1_len = np.linalg.norm(surf_struc.lattice.matrix[0]) + 2 * max_rad
                                    vec2_len = np.linalg.norm(surf_struc.lattice.matrix[1]) + 2 * max_rad
                                    vec_ang = angle(surf_struc.lattice.matrix[0], surf_struc.lattice.matrix[1])
                                    surf_cube_vol = np.sin(vec_ang) * vec1_len * vec2_len * z_len
                                    surf_tot_vol = 0
                                    for i_sp in surf_struc.species:
                                        surf_tot_vol += sphe_vol(rad_dic[str(i_sp)])

                                    int_OL_vol = Mont_geo(OL_Input[0], OL_Input[1])
                                    if int_OL_vol <= 0.01:
                                        int_OL_vol = 0

                                    rel_int_OL = int_OL_vol / surf_tot_vol
                                    un_occ_vol = surf_cube_vol - (surf_tot_vol - int_OL_vol)
                                    rel_empt_sp = un_occ_vol / surf_cube_vol
                                    Empt_score = 10
                                    Score_array[ii, jj, kk] = (1 * rel_int_OL) ** 2 + Empt_score * rel_empt_sp

  


if __name__ == "__main__":
    main()
