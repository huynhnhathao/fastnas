from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union
from nats_bench import create

_OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

class NATSAPI:
    def __init__(self, data_path: str  ) -> None:
        """
        Interface to interact with the NatsBench dataset
        
        Args:
            data_path: path to the nats data directory
        """
        self.api = create(data_path, 'tss', True, False)

    def __len__(self) -> int:
        return len(self.api)

    def genome_to_genotypes(self, genome: List[int]) -> List[int]:
        """
        decode genome to genotypes

        example genome: [1, 2, 3, 3, 4, 4] -> [(('skip_connect', 0),),
                                            (('nor_conv_1x1', 0), ('nor_conv_3x3', 1)),
                                            (('nor_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2))]

        """
        # there are 6 edges in each cell
        if len(genome) != 6:
            raise ValueError('Len of genome must be 6 for the NATS bench dataset')

        for gen in genome:
            if 4 < gen < 0:
                raise ValueError('there are only 5 operations index in [0, 4]')

        genotypes = [((_OPERATIONS[genome[0]], 0),  ), ( (_OPERATIONS[genome[1]], 0), (_OPERATIONS[genome[2]], 1) ), 
                    ( (_OPERATIONS[genome[3]], 0), (_OPERATIONS[genome[4]], 1), (_OPERATIONS[genome[5]], 2))]
        return genotypes

        
    def query_performance(self, genome: List[int], dataset: str = 'cifar100',
                      ) -> Tuple[Dict[str, float]]:

        """query the performance of the architecture constructed by the genome

        Args:
            genome: can be int or list of int. If it is an int, then query the 
                performance of the network indexed by that int. If it is a list
                of int, then decode the list into the unique string, and then to
                the network index to query.

            dataset: the dataset to query performance on. 
        """

        network_index = None
        if isinstance(genome, int):
            network_index = genome
        else:
            genotypes = self.genome_to_genotypes(genome)
            network_string = Structure(genotypes).tostr()
            # TODO: defind a method to create string structure of a given genotype
            # print(network_string)
            network_index = self.api.query_index_by_arch(network_string)
            # print(network_index)
        # print(network_index)
        cost_info = self.api.get_cost_info(network_index, dataset)
        
        more_info = self.api.get_more_info(network_index, dataset = dataset, hp = '200')
        # print(more_info)
        performance = None
        if dataset != 'cifar10':
            performance = {'FLOPs': cost_info['flops'], 'valid-accuracy': more_info['valid-accuracy'],
                            'test-accuracy': more_info['test-accuracy']}
        # no valid acc for cifar10 dataset
        else:
            performance = {'FLOPs': cost_info['flops'], 
                        'test-accuracy': more_info['test-accuracy']}

        return performance


class Structure:
    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(
            genotype, tuple
        ), "invalid class of genotype : {:}".format(type(genotype))
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, list) or isinstance(
                node_info, tuple
            ), "invalid class of node_info : {:}".format(type(node_info))
            assert len(node_info) >= 1, "invalid length : {:}".format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(
                    node_in, tuple
                ), "invalid class of in-node : {:}".format(type(node_in))
                assert (
                    len(node_in) == 2 and node_in[1] <= idx
                ), "invalid in-node : {:}".format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))

    def tolist(self, remove_str):
        # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
        # note that we re-order the input node in this function
        # return the-genotype-list and success [if unsuccess, it is not a connectivity]
        genotypes = []
        for node_info in self.nodes:
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0:
                return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), "invalid index={:} < {:}".format(
            index, len(self)
        )
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = "|".join([x[0] + "~{:}".format(x[1]) for x in node_info])
            string = "|{:}|".format(string)
            strings.append(string)
        return "+".join(strings)

    def check_valid(self):
        nodes = {0: True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == "none" or nodes[xin] is False:
                    x = False
                else:
                    x = True
                sums.append(x)
            nodes[i + 1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self, consider_zero=False):
        # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
        # two operations are special, i.e., none and skip_connect
        nodes = {0: "0"}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if consider_zero is None:
                    x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                elif consider_zero:
                    if op == "none" or nodes[xin] == "#":
                        x = "#"  # zero
                    elif op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                else:
                    if op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                cur_node.append(x)
            nodes[i_node + 1] = "+".join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_names):
        for node_info in self.nodes:
            for inode_edge in node_info:
                # assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
                if inode_edge[0] not in op_names:
                    return False
        return True

    def __repr__(self):
        return "{name}({node_num} nodes with {node_info})".format(
            name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__
        )

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        if isinstance(xstr, Structure):
            return xstr
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(
            type(xstr)
        )
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        return Structure(genotypes)

    @staticmethod
    def str2fullstructure(xstr, default_name="none"):
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(
            type(xstr)
        )
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = list((op, int(IDX)) for (op, IDX) in inputs)
            all_in_nodes = list(x[1] for x in input_infos)
            for j in range(i):
                if j not in all_in_nodes:
                    input_infos.append((default_name, j))
            node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
            genotypes.append(tuple(node_info))
        return Structure(genotypes)

    @staticmethod
    def gen_all(search_space, num, return_ori):
        assert isinstance(search_space, list) or isinstance(
            search_space, tuple
        ), "invalid class of search-space : {:}".format(type(search_space))
        assert (
            num >= 2
        ), "There should be at least two nodes in a neural cell instead of {:}".format(
            num
        )
        all_archs = get_combination(search_space, 1)
        for i, arch in enumerate(all_archs):
            all_archs[i] = [tuple(arch)]

        for inode in range(2, num):
            cur_nodes = get_combination(search_space, inode)
            new_all_archs = []
            for previous_arch in all_archs:
                for cur_node in cur_nodes:
                    new_all_archs.append(previous_arch + [tuple(cur_node)])
            all_archs = new_all_archs
        if return_ori:
            return all_archs
        else:
            return [Structure(x) for x in all_archs]

