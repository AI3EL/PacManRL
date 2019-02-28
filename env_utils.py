# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:26:23 2019

@author: jacqu

Useful functions to compute observations in the environment: 
    - direction to closest target for pacman 
"""
import numpy as np
import queue

#############################################
### Auxiliary functions for get_direction ###
#############################################

def voisins(position, game):
    """ auxiliary function for BFS. WARNING: WORKS ONLY FOR SQUARE MAPS ???!!! """
    i = position[0]
    j = position[1]
    n = game.shape[0]
    neighbours = []
    
    if i<n-1 and game[i+1][j] != 1:
        neighbours.append(((i+1, j), 'down'))
    if i>0 and game[i-1][j] != 1:
        neighbours.append(((i-1, j), 'up'))
    if j<n-1 and game[i][j+1] != 1:
        neighbours.append(((i, j+1), 'right'))
    if j>0 and game[i][j-1] != 1:
        neighbours.append(((i, j-1), 'left'))
        
    return neighbours
  
def retrace(visited, pos):
    """ auxiliary function for BFS"""
    last_move = visited[pos]
    if last_move:
        if last_move == 'up':
            pos = (pos[0]+1, pos[1])
            return retrace(visited, pos)+[1]
        if last_move == 'down':
            pos = (pos[0]-1, pos[1])
            return retrace(visited, pos)+[2]      
        if last_move == 'right':
            pos = (pos[0], pos[1]-1)
            return retrace(visited, pos)+[3]     
        if last_move == 'left':
            pos = (pos[0], pos[1]+1)
            return retrace(visited, pos)+[0]
    return []

def BFS(pacmap, pacpos):
    """ takes as input the map (self.map) and pac position, and returns a list of directions (type string)
    to reach the closest target. Uses BFS """
    pac_position = pacpos
    game = pacmap
    
    q = queue.Queue()
    q.put(pac_position)
    visited = {}
    visited[pac_position] = None
    
    while not q.empty():
        current = q.get()

        if game[current[0]][current[1]] == 2: #small ball
            a =  retrace(visited, current)
            longueur = len(a)
            a = a + [-1 for i in range(20-longueur)]
            return a
        
        neighbours = voisins(current, game)
        
        for nei in neighbours:
            if not nei[0] in visited:
                q.put(nei[0])
                visited[nei[0]]= nei[1]
    return [-1 for i in range(20)]
        

#############################################
#####        Major utils functions      #####
#############################################

def get_direction(pacmap, pacpos, gpos):
    """ returns the direction (int, 0 left, 1 down, 2 right, 3 up) to go to target
        Modify to take into account ghosts as targets !!! """
    sequence=BFS(pacmap, pacpos)
    if (sequence[0]=="left"):
        return 0
    elif(sequence[0]=="down"):
        return 1
    elif (sequence[0]=="right"):
        return 2
    else: 
        return 3
    
    # TODO: manage the case where ghosts are edible: they become an prioritary target
    

def ghost_threats(pacmap, pacpos, gpos, gdir):
    """ returns an array of 4 int (values are 0 or 1) indicating the presence of a ghost threat in each direction"""
    
    # now we evaluate the presence of ghost threat in each direction (only ones with no wall...)
    s=np.zeros(4)
    for gp, g_dir in zip(gpos, gdir):
        i,j= gp[0],gp[1]
        if(i==pacpos[0]):
            # there is a ghost on same line
            if(j>pacpos[1]):
                # the ghost is on the right of pac
                path=pacmap[i][pacpos[1]:j]
                if(1 not in path):
                    # then there is no wall to protect pac
                    if(g_dir==0): # ghost is moving left, towards pac
                        s[7-5]=1        
            else:
                # the ghost is on the left of pac
                path=pacmap[i][j:pacpos[1]]
                if(1 not in path):
                    # then there is no wall to protect pac
                    if(g_dir==2): # ghost is moving right, towards pac
                        s[5-5]=1
                
        if(j==pacpos[1]):
            # there is a ghost on same column
            if(i>pacpos[1]):
                # the ghost is under pac
                path=pacmap[pacpos[1]:i][j]
                if(1 not in path):
                    # then there is no wall to protect pac
                    if(g_dir==3): # ghost is moving up, towards pac
                        s[6-5]=1
            else:
                # the ghost is  above pac
                path=pacmap[i:pacpos[1]][j]
                if(1 not in path):
                    # then there is no wall to protect pac
                    if(g_dir==1): # ghost is moving down, towards pac
                        s[8-5]=1
                        
    return s