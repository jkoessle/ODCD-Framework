from pm4py import discover_dfg_typed


def check_dfg_graph_freq(log):
    graph, sa, ea = discover_dfg_typed(log)
    
    freq = sum(graph.values())
    
    return freq