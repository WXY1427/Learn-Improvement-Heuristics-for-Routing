
# coding: utf-8

# In[6]:


from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import torch
from options import get_options
from utils import load_problem
from torch.utils.data import DataLoader
import math
import numpy as np
import time

val_dataset = torch.load('myval_200_2.pt')


# In[7]:


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)**0.5


# In[8]:


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Distance callback
def create_distance_callback(dist_matrix):
  # Create a callback to calculate distances between cities.

  def distance_callback(from_node, to_node):
    return int(dist_matrix[from_node][to_node])

  return distance_callback

def run(dis):
    # Cities
#     city_names = ["1", "2", "3", "4", "5", "6", "7",
#                 "8", "9", "10", "11", "12"]
    
    city_names = [str(x) for x in list(range(1, 201))]
    
    # Distance matrix
    dist_matrix = dis
    
    tsp_size = len(city_names)
    num_routes = 1
    depot = 0
    
    # Create routing model
    if tsp_size > 0:
            routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
            search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        # Create the distance callback.
            dist_callback = create_distance_callback(dist_matrix)
            routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
            
        # Solve the problem.
            assignment = routing.SolveWithParameters(search_parameters)

            if assignment:
          # Solution distance.
                #print("Total distance:" + str(assignment.ObjectiveValue()/10000))
          # Display the solution.
          # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
                route_number = 0
                index = routing.Start(route_number) # Index of the variable for the starting node.
                route = ''
                while not routing.IsEnd(index):
                  # Convert variable indices to node indices in the displayed route.
                    route += str(city_names[routing.IndexToNode(index)]) + ' -> '
                    index = assignment.Value(routing.NextVar(index))
                    route += str(city_names[routing.IndexToNode(index)])
#                     node_index = routing.IndexToNode(index)
#                     next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
#                     route_dist += routing.GetArcCostForVehicle(node_index, next_node_index, 0)
#                     print("Route:\n\n" + route)
            else:
                print('No solution found.')
            
    else:
        print('Specify an instance greater than 0.')
    
    return assignment.ObjectiveValue()/10000

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances

def print_solution(manager, routing, solution):
    """Prints solution on console."""
#    print('Objective: {}'.format(solution.ObjectiveValue()/10000))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
#    print(plan_output)
    plan_output += 'Objective: {}m\n'.format(route_distance)
    return solution.ObjectiveValue()/100000


# In[9]:

tt= time.time()
l_dis = []
for i in range(500):
    cor_with_depot = val_dataset[i]
    dis_matrix = pairwise_distances(cor_with_depot)
    dis_matrix_int = np.round((dis_matrix*100000).numpy()).astype(np.int)

    data = {}
    data['num_vehicles'] = 1
    data['depot'] = 0
    data['locations'] = val_dataset[i]*100000


    """Entry point of the program."""

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'])

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        l_dis.append(print_solution(manager, routing, solution))

print('Best solution: {} +- {}'.format(
    np.mean(l_dis), np.std(l_dis) / math.sqrt(len(l_dis))))
print(time.time()-tt)

