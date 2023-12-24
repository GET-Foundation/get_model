from typing import Tuple, List, Dict

import pandas as pd

def cell_splitter(celltype_metadata: pd.DataFrame, leave_out_celltypes: str, datatypes: str, is_train: bool = True, is_pretrain: bool = False) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Process data based on given parameters.

    Args:
        celltype_metadata (str): Path to the cell type metadata file in CSV format.
        leave_out_celltypes (str): Comma-separated string of cell types to be excluded or used for validation.
        datatypes (str): Comma-separated string of data types to be considered.
        is_train (bool, optional): Specify if the processing is for training data. Defaults to True.
        is_pretrain (bool, optional): Specify if the processing is for pre-training data. Defaults to False.

    Returns:
        Tuple[List[str], Dict[str, str], Dict[str, str]]: A tuple containing the list of target file IDs, cell labels dictionary,
        and datatype dictionary.

    Raises:
        AssertionError: If the length of file_list, cluster_list, datatype_list, and expression_list is not equal.

    """
    # Convert leave_out_celltypes, datatypes, and leave_out_chromosomes to lists
    leave_out_celltypes = leave_out_celltypes.split(",")
    datatypes = datatypes.split(",")

    # Process cell types
    celltype_list = sorted(celltype_metadata["cluster"].unique().tolist())
    if is_train:
        for cell in leave_out_celltypes:
            if cell in celltype_list:
                celltype_list.remove(cell)
        print("train cell types list:", celltype_list)
        print("train data types list:", datatypes)
    else:
        if leave_out_celltypes == [""]:
            pass
        else:
            celltype_list = leave_out_celltypes
        print("val cell types list:", leave_out_celltypes)
        print("val data types list:", datatypes)

    # Initialize variables
    file_id_list = []
    datatype_dict = {}
    cell_dict = {}


    # Iterate over cell types
    for cell in celltype_list:
        celltype_metadata_of_cell = celltype_metadata.loc[celltype_metadata["cluster"] == cell]
        file_list = celltype_metadata_of_cell["id"].tolist()
        cluster_list = celltype_metadata_of_cell["cluster"].tolist()
        datatype_list = celltype_metadata_of_cell["datatype"].tolist()
        expression_list = celltype_metadata_of_cell["expression"].tolist()

        assert len(file_list) == len(cluster_list) == len(datatype_list) == len(expression_list)

        # Process each file
        for file, cluster, datatype, expression in zip(file_list, cluster_list, datatype_list, expression_list):
            if is_pretrain and datatype in datatypes: # load cell types with ATAC but no expression
                file_id_list.append(file)
                cell_dict[file] = cluster
                datatype_dict[file] = datatype
            elif datatype in datatypes and expression == 'True':
                file_id_list.append(file)
                cell_dict[file] = cluster
                datatype_dict[file] = datatype
            else:
                continue

    if not is_train:
        file_id_list = sorted(file_id_list)

    print("file id list: ", file_id_list)
    return file_id_list, cell_dict, datatype_dict


def chromosome_splitter(all_chromosomes: List[str], leave_out_chromosomes: List[str], is_train: bool = True) -> List[str]:
    """
    Process chromosomes based on given parameters.

    Args:
        all_chromosomes (List[str]): List of all available chromosomes.
        leave_out_chromosomes (List[str]): List of chromosomes to be excluded.
        is_train (bool, optional): Specify if the processing is for training data. Defaults to True.

    Returns:
        List[str]: List of input chromosomes.

    """
    # Copy the all_chromosomes list to input_chromosomes
    input_chromosomes = all_chromosomes.copy()
    leave_out_chromosomes = leave_out_chromosomes.split(",")
    if is_train:
        for leave_out_chromosome in leave_out_chromosomes:
            if leave_out_chromosome in all_chromosomes:
                input_chromosomes.remove(leave_out_chromosome)
    else:
        if leave_out_chromosomes == [""]:
            input_chromosomes = all_chromosomes
        else:
            input_chromosomes = leave_out_chromosomes

    if isinstance(input_chromosomes, str):
        input_chromosomes = [input_chromosomes]
    
    print("input chromosomes: ", input_chromosomes)
    return input_chromosomes

