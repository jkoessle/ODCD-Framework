'''
this is a modified version ofthe file taken from the pm4py library
you can view the whole library this link:
https://github.com/pm4py

author of the modifications:
Anton Yeshchenko
'''
import math
import tempfile
import pydotplus
from pm4py.visualization.common.utils import human_readable_stat

color_for_constraints = {}

color_for_constraints['Succession'] = '#00b8bd'
color_for_constraints['Precedence'] = '#00b8bd'
color_for_constraints['Response'] = '#00b8bd'

color_for_constraints['ChainSuccession'] = '#06d81f'
color_for_constraints['ChainPrecedence'] = '#06d81f'
color_for_constraints['ChainResponse'] = '#06d81f'

color_for_constraints['NotSuccession'] = '#ff0000'
color_for_constraints['NotChainSuccession'] = '#ff0000'

constraints_to_draw = ['Succession','Response','Precedence',
                      'ChainSuccession', 'ChainPrecedence', 'ChainResponse',
                       'NotChainSuccession','NotSuccession']

from types import MethodType

def get_corr_hex(num):
    """
    Gets correspondence between a number
    and an hexadecimal string

    Parameters
    -------------
    num
        Number

    Returns
    -------------
    hex_string
        Hexadecimal string
    """
    if num < 10:
        return str(int(num))
    elif num < 11:
        return "A"
    elif num < 12:
        return "B"
    elif num < 13:
        return "C"
    elif num < 14:
        return "D"
    elif num < 15:
        return "E"
    elif num < 16:
        return "F"


def transform_to_hex(graycolor):
    """
    Transform color to hexadecimal representation

    Parameters
    -------------
    graycolor
        Gray color (int from 0 to 255)

    Returns
    -------------
    hex_string
        Hexadecimal color
    """
    left0 = graycolor / 16
    right0 = graycolor % 16

    left00 = get_corr_hex(left0)
    right00 = get_corr_hex(right0)

    return "#" + left00 + right00 + left00 + right00 + left00 + right00


def transform_to_hex_2(color):
    """
    Transform color to hexadecimal representation

    Parameters
    -------------
    color
        Gray color (int from 0 to 255)

    Returns
    -------------
    hex_string
        Hexadecimal color
    """
    color = 255 - color
    color2 = 255 - color

    left0 = color / 16
    right0 = color % 16

    left1 = color2 / 16
    right1 = color2 % 16

    left0 = get_corr_hex(left0)
    right0 = get_corr_hex(right0)
    left1 = get_corr_hex(left1)
    right1 = get_corr_hex(right1)

    return "#" + left0 + right0 + left1 + right1 + left1 + right1

def apply(heu_net, parameters=None):
    """
    Gets a representation of an Heuristics Net

    Parameters
    -------------
    heu_net
        Heuristics net
    parameters
        Possible parameters of the algorithm, including: format

    Returns
    ------------
    gviz
        Representation of the Heuristics Net
    """
    if parameters is None:
        parameters = {}

    image_format = parameters["format"] if "format" in parameters else "png"

    graph = pydotplus.Dot(strict=True)
    graph.obj_dict['attributes']['bgcolor'] = 'transparent'

    corr_nodes = {}
    corr_nodes_names = {}
    is_frequency = False

    for node_name in heu_net.nodes:
        node = heu_net.nodes[node_name]
        node_occ = node.node_occ
        graycolor = transform_to_hex_2(max(255 - math.log(node_occ) * 9, 0))
        if node.node_type == "frequency":
            is_frequency = True
            n = pydotplus.Node(name=node_name, shape="box", style="filled",
                               label=node_name + " (" + str(node_occ) + ")", fillcolor=graycolor)
        else:
            n = pydotplus.Node(name=node_name, shape="box", style="filled",
                               label=node_name, fillcolor=graycolor)
        corr_nodes[node] = n
        corr_nodes_names[node_name] = n
        graph.add_node(n)

    # gets max arc value
    max_arc_value = -1
    for node_name in heu_net.nodes:
        node = heu_net.nodes[node_name]
        for other_node in node.output_connections:
            if other_node in corr_nodes:
                for edge in node.output_connections[other_node]:
                    max_arc_value = max(max_arc_value, edge.repr_value)

    for node_name in heu_net.nodes:
        node = heu_net.nodes[node_name]
        for other_node in node.output_connections:
            if other_node in corr_nodes:
                for edge in node.output_connections[other_node]:
                    this_pen_width = 1.0 + math.log(1 + edge.repr_value) / 11.0
                    repr_value = str(edge.repr_value)
                    if edge.net_name:
                        if node.node_type == "frequency":
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node],
                                               label=edge.net_name + " (" + repr_value + ")",
                                               color=edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node],
                                               label=edge.net_name + " (" + human_readable_stat(repr_value) + ")",
                                               color=edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)
                    else:  # this is where the arcs are drown.
                        if node.node_type == "frequency":
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node], label=repr_value,
                                               color= '#de3c59',
#                                               edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node],
                                               label=human_readable_stat(repr_value),
                                               color=edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)

                    graph.add_edge(e)

    for index, sa_list in enumerate(heu_net.start_activities):
        effective_sa_list = [n for n in sa_list if n in corr_nodes_names]
        if effective_sa_list:
            start_i = pydotplus.Node(name="start_" + str(index), label="@@S", color=heu_net.default_edges_color[index],
                                     fontsize="8", fontcolor="#32CD32", fillcolor="#32CD32",
                                     style="filled")
            graph.add_node(start_i)
            for node_name in effective_sa_list:
                sa = corr_nodes_names[node_name]
                if type(heu_net.start_activities[index]) is dict:
                    if is_frequency:
                        occ = heu_net.start_activities[index][node_name]
                        this_pen_width = 1.0 + math.log(1 + occ) / 11.0
                        if heu_net.net_name[index]:
                            e = pydotplus.Edge(src=start_i, dst=sa, label=heu_net.net_name[index] + " (" + str(occ) + ")",
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=start_i, dst=sa, label=str(occ),
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                    else:
                        e = pydotplus.Edge(src=start_i, dst=sa, label=heu_net.net_name[index],
                                           color=heu_net.default_edges_color[index],
                                           fontcolor=heu_net.default_edges_color[index])
                else:
                    e = pydotplus.Edge(src=start_i, dst=sa, label=heu_net.net_name[index],
                                       color=heu_net.default_edges_color[index],
                                       fontcolor=heu_net.default_edges_color[index])
                graph.add_edge(e)

    for index, ea_list in enumerate(heu_net.end_activities):
        effective_ea_list = [n for n in ea_list if n in corr_nodes_names]
        if effective_ea_list:
            end_i = pydotplus.Node(name="end_" + str(index), label="@@E", color="#",
                                   fillcolor="#FFA500", fontcolor="#FFA500", fontsize="8",
                                   style="filled")
            graph.add_node(end_i)
            for node_name in effective_ea_list:
                ea = corr_nodes_names[node_name]
                if type(heu_net.end_activities[index]) is dict:
                    if is_frequency:
                        occ = heu_net.end_activities[index][node_name]
                        this_pen_width = 1.0 + math.log(1 + occ) / 11.0
                        if heu_net.net_name[index]:
                            e = pydotplus.Edge(src=ea, dst=end_i, label=heu_net.net_name[index] + " (" + str(occ) + ")",
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=ea, dst=end_i, label=str(occ),
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                    else:
                        e = pydotplus.Edge(src=ea, dst=end_i, label=heu_net.net_name[index],
                                           color=heu_net.default_edges_color[index],
                                           fontcolor=heu_net.default_edges_color[index])
                else:
                    e = pydotplus.Edge(src=ea, dst=end_i, label=heu_net.net_name[index],
                                       color=heu_net.default_edges_color[index],
                                       fontcolor=heu_net.default_edges_color[index])
                graph.add_edge(e)

    file_name = tempfile.NamedTemporaryFile(suffix='.' + image_format)
    file_name.close()
    graph.write(file_name.name, format=image_format)
    return file_name


def apply_with_constraints(heu_net, parameters=None, constr = None):
    """
    Gets a representation of an Heuristics Net

    Parameters
    -------------
    heu_net
        Heuristics net
    parameters
        Possible parameters of the algorithm, including: format

    Returns
    ------------
    gviz
        Representation of the Heuristics Net
    """
    if parameters is None:
        parameters = {}

    image_format = parameters["format"] if "format" in parameters else "png"

    graph = pydotplus.Dot(strict=True)
    graph.obj_dict['attributes']['bgcolor'] = 'transparent'

    corr_nodes = {}
    corr_nodes_names = {}
    is_frequency = False

    for node_name in heu_net.nodes:
        node = heu_net.nodes[node_name]
        node_occ = node.node_occ
        graycolor = transform_to_hex_2(max(255 - math.log(node_occ) * 9, 0))
        if node.node_type == "frequency":
            is_frequency = True
            n = pydotplus.Node(name=node_name, shape="box", style="filled",
                               label=node_name + " (" + str(node_occ) + ")", fillcolor=graycolor)
        else:
            n = pydotplus.Node(name=node_name, shape="box", style="filled",
                               label=node_name, fillcolor=graycolor)
        corr_nodes[node] = n
        corr_nodes_names[node_name] = n
        graph.add_node(n)

    # gets max arc value
    max_arc_value = -1
    for node_name in heu_net.nodes:
        node = heu_net.nodes[node_name]
        for other_node in node.output_connections:
            if other_node in corr_nodes:
                for edge in node.output_connections[other_node]:
                    max_arc_value = max(max_arc_value, edge.repr_value)

    for node_name in heu_net.nodes:
        node = heu_net.nodes[node_name]
        for other_node in node.output_connections:
            if other_node in corr_nodes:
                for edge in node.output_connections[other_node]:
                    this_pen_width = 1.0 + math.log(1 + edge.repr_value) / 11.0
                    repr_value = str(edge.repr_value)
                    if edge.net_name:
                        if node.node_type == "frequency":
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node],
                                               label=edge.net_name + " (" + repr_value + ")",
                                               color=edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node],
                                               label=edge.net_name + " (" + human_readable_stat(repr_value) + ")",
                                               color=edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)
                    else:  # this is where the arcs are drown.
                        if node.node_type == "frequency":

                            ########################
                            # drawing other colored arcs for the constraints
                            source_node = corr_nodes[other_node].obj_dict['name'][1:-1]
                            destination_node = corr_nodes[node].obj_dict['name'][1:-1]

                            #print(source_node + '   ' + destination_node + '  ' + str(repr_value))
                            color_arc = edge.repr_color

                            for con in constraints_to_draw:
                                if con in constr and (destination_node, source_node) in constr[con]:
                                    color_arc = color_for_constraints[con]
                                    constr[con].remove((destination_node, source_node))
                                    repr_value += ' ' + con

                            # if  'Succession' in constr and (destination_node,source_node) in constr['Succession']:
                            #     color_arc = color_for_constraints['Succession']
                            #     constr['Succession'].remove((destination_node,source_node))
                            # elif(destination_node,source_node) in constr['NotSuccession']:
                            #     color_arc = color_for_constraints['NotSuccession']
                            #     constr['NotSuccession'].remove((destination_node, source_node))
                            ########################
                            ########################



                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node], label=repr_value,
                                               color=color_arc,
                                               fontcolor=color_arc, penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=corr_nodes[node], dst=corr_nodes[other_node],
                                               label=human_readable_stat(repr_value),
                                               color=edge.repr_color,
                                               fontcolor=edge.repr_color, penwidth=this_pen_width)

                    graph.add_edge(e)

########################
########################
    # add addditonaal edges that do not exist naturally, ro represent the missing constraints
    def draw_additional(graph, constr = set() ,  constr_name = 'Succession'):
        if not constr_name in constr:
            return
        for con in list(constr[constr_name]):
            src = (con[0])
            dst = (con[1])
            e = pydotplus.Edge(src=src, dst=dst, label=constr_name,
                               color=color_for_constraints[constr_name],
                               fontcolor=color_for_constraints[constr_name], penwidth=1)

            graph.add_edge(e)

    for con in constraints_to_draw:
        draw_additional(graph, constr, con)
    # draw_additional(graph, constr, 'NotSuccession')


########################


    for index, sa_list in enumerate(heu_net.start_activities):
        effective_sa_list = [n for n in sa_list if n in corr_nodes_names]
        if effective_sa_list:
            start_i = pydotplus.Node(name="start_" + str(index), label="@@S", color=heu_net.default_edges_color[index],
                                     fontsize="8", fontcolor="#32CD32", fillcolor="#32CD32",
                                     style="filled")
            graph.add_node(start_i)
            for node_name in effective_sa_list:
                sa = corr_nodes_names[node_name]
                if type(heu_net.start_activities[index]) is dict:
                    if is_frequency:
                        occ = heu_net.start_activities[index][node_name]
                        this_pen_width = 1.0 + math.log(1 + occ) / 11.0
                        if heu_net.net_name[index]:
                            e = pydotplus.Edge(src=start_i, dst=sa, label=heu_net.net_name[index] + " (" + str(occ) + ")",
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=start_i, dst=sa, label=str(occ),
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                    else:
                        e = pydotplus.Edge(src=start_i, dst=sa, label=heu_net.net_name[index],
                                           color=heu_net.default_edges_color[index],
                                           fontcolor=heu_net.default_edges_color[index])
                else:
                    e = pydotplus.Edge(src=start_i, dst=sa, label=heu_net.net_name[index],
                                       color=heu_net.default_edges_color[index],
                                       fontcolor=heu_net.default_edges_color[index])
                graph.add_edge(e)

    for index, ea_list in enumerate(heu_net.end_activities):
        effective_ea_list = [n for n in ea_list if n in corr_nodes_names]
        if effective_ea_list:
            end_i = pydotplus.Node(name="end_" + str(index), label="@@E", color="#",
                                   fillcolor="#FFA500", fontcolor="#FFA500", fontsize="8",
                                   style="filled")
            graph.add_node(end_i)
            for node_name in effective_ea_list:
                ea = corr_nodes_names[node_name]
                if type(heu_net.end_activities[index]) is dict:
                    if is_frequency:
                        occ = heu_net.end_activities[index][node_name]
                        this_pen_width = 1.0 + math.log(1 + occ) / 11.0
                        if heu_net.net_name[index]:
                            e = pydotplus.Edge(src=ea, dst=end_i, label=heu_net.net_name[index] + " (" + str(occ) + ")",
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                        else:
                            e = pydotplus.Edge(src=ea, dst=end_i, label=str(occ),
                                               color=heu_net.default_edges_color[index],
                                               fontcolor=heu_net.default_edges_color[index], penwidth=this_pen_width)
                    else:
                        e = pydotplus.Edge(src=ea, dst=end_i, label=heu_net.net_name[index],
                                           color=heu_net.default_edges_color[index],
                                           fontcolor=heu_net.default_edges_color[index])
                else:
                    e = pydotplus.Edge(src=ea, dst=end_i, label=heu_net.net_name[index],
                                       color=heu_net.default_edges_color[index],
                                       fontcolor=heu_net.default_edges_color[index])
                graph.add_edge(e)

    file_name = tempfile.NamedTemporaryFile(suffix='.' + image_format)
    file_name.close()
    graph.write(file_name.name, format=image_format)
    return file_name
