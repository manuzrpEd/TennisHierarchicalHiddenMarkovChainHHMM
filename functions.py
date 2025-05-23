##### Clean functions 
## This file is structure to build Markov chains models, in this order :
# I - Game model 
# II - Tie-break model
# III - Set model
# IV - Match model

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from quantecon import MarkovChain

################### 0 - Define constants #######################################
# (a) Standard state for a game
game_states = ["0-0","0-15","15-0","15-15",
               "30-0","0-30","40-0","30-15",
               "15-30","0-40","40-15","15-40",
               "30-30(DEUCE)","40-30(A-40)","30-40(40-A)",
               "HOLD" # keeping the game/service (absorbing state)
               , "BREAK" # breaking the service (absorbing state)
               ]
s0game = pd.DataFrame(np.zeros((1, len(game_states))), columns=game_states)
s0game.at[0, "0-0"] = 1 # initializing the Markov chain in the starting state

# (b) Standard state for a tie-break
tb_states = ["0-0","0-1","1-0","1-1",
             "2-0","0-2","3-0","2-1",
             "1-2","0-3","4-0","3-1",
             "2-2","1-3","0-4","5-0",
             "4-1", "3-2","2-3","1-4",
             "0-5","5-1","4-2","3-3",
             "2-4","1-5","5-2","4-3","3-4",
             "2-5","5-3","4-4","3-5","5-4",
             "4-5", "5-5","6-5","5-6",
             "6-6"
             ,"SETv1" # absorbing state
             ,"SETv2" # absorbing state
             ,"6-0",
             "6-1","6-2","6-3","6-4","4-6",
             "3-6","2-6","1-6","0-6","7-7","7-6","6-7"]
s0tb = pd.DataFrame(np.zeros((1, len(tb_states))), columns=tb_states)
s0tb.at[0, "0-0"] = 1 # initializing the Markov chain in the starting state

# (c) Standard state for a set
set_states = ["0-0","0-1","1-0","1-1",
              "2-0","0-2","3-0","2-1",
              "1-2","0-3","4-0","3-1",
              "2-2","1-3","0-4","5-0",
              "4-1", "3-2","2-3","1-4",
              "0-5","5-1","4-2","3-3",
              "2-4","1-5","5-2","4-3","3-4",
              "2-5","5-3","4-4","3-5","5-4",
              "4-5", "5-5","6-5","5-6",
              "6-6"
              ,"SETv1" # absorbing state
              ,"SETv2" # absorbing state
              ]
s0set = pd.DataFrame(np.zeros((1, len(set_states))), columns=set_states)
s0set.at[0, "0-0"] = 1 # initializing the Markov chain in the starting state

# (d) Standard state for a match
match_states = ["0-0","0-1","1-0","1-1","2-0","0-2","2-1","1-2"
                ,"V1" # absorbing state
                ,"V2" # absorbing state
                ]
s0match = pd.DataFrame(np.zeros((1, len(match_states))), columns=match_states)

################# I - Game model  ##############################################
# (a) Build the transition matrix for a game
def MCgame2(ppoint_server):
    ppoint_ret = 1 - ppoint_server
    STATES = ["0-0","0-15","15-0","15-15",
              "30-0","0-30","40-0","30-15",
              "15-30","0-40","40-15","15-40",
              "30-30(DEUCE)","40-30(A-40)","30-40(40-A)",
              "HOLD" # absorbing state
              , "BREAK" # absorbing state
              ]
    idx = {state: i for i, state in enumerate(STATES)}
    tMat = np.zeros((len(STATES), len(STATES)))

    # Set the correct probabilities (server wins point)
    tMat[idx["0-0"], idx["15-0"]] = ppoint_server
    tMat[idx["15-0"], idx["30-0"]] = ppoint_server
    tMat[idx["0-15"], idx["15-15"]] = ppoint_server
    tMat[idx["30-0"], idx["40-0"]] = ppoint_server
    tMat[idx["15-15"], idx["30-15"]] = ppoint_server
    tMat[idx["0-30"], idx["15-30"]] = ppoint_server
    tMat[idx["40-0"], idx["HOLD"]] = ppoint_server
    tMat[idx["30-15"], idx["40-15"]] = ppoint_server
    tMat[idx["40-15"], idx["HOLD"]] = ppoint_server
    tMat[idx["40-30(A-40)"], idx["HOLD"]] = ppoint_server
    tMat[idx["0-40"], idx["15-40"]] = ppoint_server
    tMat[idx["15-40"], idx["30-40(40-A)"]] = ppoint_server
    tMat[idx["30-40(40-A)"], idx["30-30(DEUCE)"]] = ppoint_server
    tMat[idx["15-30"], idx["30-30(DEUCE)"]] = ppoint_server
    tMat[idx["30-30(DEUCE)"], idx["40-30(A-40)"]] = ppoint_server

    # Set the correct probabilities (returner wins point)
    tMat[idx["0-0"], idx["0-15"]] = ppoint_ret
    tMat[idx["15-0"], idx["15-15"]] = ppoint_ret
    tMat[idx["0-15"], idx["0-30"]] = ppoint_ret
    tMat[idx["30-0"], idx["30-15"]] = ppoint_ret
    tMat[idx["15-15"], idx["15-30"]] = ppoint_ret
    tMat[idx["0-30"], idx["0-40"]] = ppoint_ret
    tMat[idx["40-0"], idx["40-15"]] = ppoint_ret
    tMat[idx["30-15"], idx["30-30(DEUCE)"]] = ppoint_ret
    tMat[idx["40-15"], idx["40-30(A-40)"]] = ppoint_ret
    tMat[idx["40-30(A-40)"], idx["30-30(DEUCE)"]] = ppoint_ret
    tMat[idx["0-40"], idx["BREAK"]] = ppoint_ret
    tMat[idx["15-40"], idx["BREAK"]] = ppoint_ret
    tMat[idx["30-40(40-A)"], idx["BREAK"]] = ppoint_ret
    tMat[idx["15-30"], idx["15-40"]] = ppoint_ret
    tMat[idx["30-30(DEUCE)"], idx["30-40(40-A)"]] = ppoint_ret

    # Stationary (absorbing) states
    tMat[idx["HOLD"], idx["HOLD"]] = 1
    tMat[idx["BREAK"], idx["BREAK"]] = 1

    # Create the Markov Chain object
    MC_game2 = MarkovChain(tMat, STATES)
    return MC_game2

# (b) Compute outcome probabilities for a service game
def resGAME(ppoint_server, s_game, graph=False):
    MC_game1 = MCgame2(ppoint_server)
    # s_game is a 1x17 numpy array or pandas DataFrame (one-hot vector)
    # To get the distribution after many steps, multiply by the transition matrix raised to a large power
    tMat = MC_game1.P
    s_game = np.array(s_game).reshape(1, -1)
    # Matrix power
    tMat_n = np.linalg.matrix_power(tMat, 10000)
    resGAME = np.dot(s_game, tMat_n)
    # Optionally, show the chain graph
    if graph:
        # Build directed graph from transition matrix
        G = nx.DiGraph()
        states = MC_game1.state_values
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                prob = tMat[i, j]
                if prob > 0:
                    G.add_edge(from_state, to_state, weight=prob, label=f"{prob:.2f}")

        pos = nx.spring_layout(G, seed=42)  # or use nx.circular_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')
        plt.figure(figsize=(12, 5))
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Markov Chain: Tennis Game States")
        plt.axis('off')
        plt.show()
    return pd.DataFrame(resGAME, columns=MC_game1.state_values)

################## II - Tie-break model ########################################
def MCtb2(ppoint_srv1, ppoint_srv2):
    # Define the states
    STATES = [
        "0-0","0-1","1-0","1-1",
        "2-0","0-2","3-0","2-1",
        "1-2","0-3","4-0","3-1",
        "2-2","1-3","0-4","5-0",
        "4-1", "3-2","2-3","1-4",
        "0-5","5-1","4-2","3-3",
        "2-4","1-5","5-2","4-3","3-4",
        "2-5","5-3","4-4","3-5","5-4",
        "4-5", "5-5","6-5","5-6",
        "6-6"
        ,"SETv1" # absorbing state
        ,"SETv2" # absorbing state
        ,"6-0",
        "6-1","6-2","6-3","6-4","4-6",
        "3-6","2-6","1-6","0-6","7-7","7-6","6-7"
    ]
    idx = {state: i for i, state in enumerate(STATES)}
    tMat = np.zeros((len(STATES), len(STATES)))

    # Helper for setting transitions
    def set_trans(from_state, to_state, prob):
        tMat[idx[from_state], idx[to_state]] = prob

    # Fill transitions as per R code
    # Player 1 serving
    set_trans("0-0","1-0", ppoint_srv1)
    set_trans("3-0","4-0", ppoint_srv1)
    set_trans("2-1","3-1", ppoint_srv1)
    set_trans("1-2","2-2", ppoint_srv1)
    set_trans("0-3","1-3", ppoint_srv1)
    set_trans("4-0","5-0", ppoint_srv1)
    set_trans("3-1","4-1", ppoint_srv1)
    set_trans("2-2","3-2", ppoint_srv1)
    set_trans("1-3","2-3", ppoint_srv1)
    set_trans("0-4","1-4", ppoint_srv1)
    set_trans("6-1","SETv1", ppoint_srv1)
    set_trans("5-2","6-2", ppoint_srv1)
    set_trans("4-3","5-3", ppoint_srv1)
    set_trans("3-4","4-4", ppoint_srv1)
    set_trans("2-5","3-5", ppoint_srv1)
    set_trans("1-6","2-6", ppoint_srv1)
    set_trans("6-2","SETv1", ppoint_srv1)
    set_trans("5-3","6-3", ppoint_srv1)
    set_trans("4-4","5-4", ppoint_srv1)
    set_trans("3-5","4-5", ppoint_srv1)
    set_trans("2-6","3-6", ppoint_srv1)
    set_trans("6-5","SETv1", ppoint_srv1)
    set_trans("5-6","6-6", ppoint_srv1)
    set_trans("6-6","7-6", ppoint_srv1)

    set_trans("0-0","0-1", 1-ppoint_srv1)
    set_trans("3-0","3-1", 1-ppoint_srv1)
    set_trans("2-1","2-2", 1-ppoint_srv1)
    set_trans("1-2","1-3", 1-ppoint_srv1)
    set_trans("0-3","0-4", 1-ppoint_srv1)
    set_trans("4-0","4-1", 1-ppoint_srv1)
    set_trans("3-1","3-2", 1-ppoint_srv1)
    set_trans("2-2","2-3", 1-ppoint_srv1)
    set_trans("1-3","1-4", 1-ppoint_srv1)
    set_trans("0-4","0-5", 1-ppoint_srv1)
    set_trans("6-1","6-2", 1-ppoint_srv1)
    set_trans("5-2","5-3", 1-ppoint_srv1)
    set_trans("4-3","4-4", 1-ppoint_srv1)
    set_trans("3-4","3-5", 1-ppoint_srv1)
    set_trans("2-5","2-6", 1-ppoint_srv1)
    set_trans("1-6","SETv2", 1-ppoint_srv1)
    set_trans("6-2","6-3", 1-ppoint_srv1)
    set_trans("5-3","5-4", 1-ppoint_srv1)
    set_trans("4-4","4-5", 1-ppoint_srv1)
    set_trans("3-5","3-6", 1-ppoint_srv1)
    set_trans("2-6","SETv2", 1-ppoint_srv1)
    set_trans("6-5","6-6", 1-ppoint_srv1)
    set_trans("5-6","SETv2", 1-ppoint_srv1)
    set_trans("3-4","3-5", 1-ppoint_srv1)
    set_trans("6-6","6-7", 1-ppoint_srv1)

    # Player 2 serving
    set_trans("1-0","1-1", ppoint_srv2)
    set_trans("0-1","0-2", ppoint_srv2)
    set_trans("2-0","2-1", ppoint_srv2)
    set_trans("1-1","1-2", ppoint_srv2)
    set_trans("0-2","0-3", ppoint_srv2)
    set_trans("0-3","0-4", ppoint_srv2)
    set_trans("5-0","5-1", ppoint_srv2)
    set_trans("4-1","4-2", ppoint_srv2)
    set_trans("3-2","3-3", ppoint_srv2)
    set_trans("2-3","2-4", ppoint_srv2)
    set_trans("1-4","1-5", ppoint_srv2)
    set_trans("0-5","0-6", ppoint_srv2)
    set_trans("6-0","6-1", ppoint_srv2)
    set_trans("5-1","5-2", ppoint_srv2)
    set_trans("4-2","5-2", ppoint_srv2)
    set_trans("3-3","3-4", ppoint_srv2)
    set_trans("2-4","2-5", ppoint_srv2)
    set_trans("1-5","1-6", ppoint_srv2)
    set_trans("0-6","SETv2", ppoint_srv2)
    set_trans("6-3","6-4", ppoint_srv2)
    set_trans("5-4","5-5", ppoint_srv2)
    set_trans("4-5","4-6", ppoint_srv2)
    set_trans("3-6","SETv2", ppoint_srv2)
    set_trans("6-4","6-5", ppoint_srv2)
    set_trans("5-5","5-6", ppoint_srv2)
    set_trans("4-6","SETv2", ppoint_srv2)
    set_trans("6-7","SETv2", ppoint_srv2)
    set_trans("7-6","7-7", ppoint_srv2)
    set_trans("4-2","4-3", ppoint_srv2)
    set_trans("7-7","5-6", ppoint_srv2)

    set_trans("1-0","2-0", 1-ppoint_srv2)
    set_trans("0-1","1-1", 1-ppoint_srv2)
    set_trans("2-0","3-0", 1-ppoint_srv2)
    set_trans("1-1","2-1", 1-ppoint_srv2)
    set_trans("0-2","1-2", 1-ppoint_srv2)
    set_trans("0-3","1-3", 1-ppoint_srv2)
    set_trans("5-0","6-0", 1-ppoint_srv2)
    set_trans("4-1","5-1", 1-ppoint_srv2)
    set_trans("3-2","4-2", 1-ppoint_srv2)
    set_trans("2-3","3-3", 1-ppoint_srv2)
    set_trans("1-4","2-4", 1-ppoint_srv2)
    set_trans("0-5","1-5", 1-ppoint_srv2)
    set_trans("6-0","SETv1", 1-ppoint_srv2)
    set_trans("5-1","6-1", 1-ppoint_srv2)
    set_trans("4-2","5-2", 1-ppoint_srv2)
    set_trans("3-3","4-3", 1-ppoint_srv2)
    set_trans("2-4","3-4", 1-ppoint_srv2)
    set_trans("1-5","2-5", 1-ppoint_srv2)
    set_trans("0-6","1-6", 1-ppoint_srv2)
    set_trans("6-3","SETv1", 1-ppoint_srv2)
    set_trans("5-4","6-4", 1-ppoint_srv2)
    set_trans("4-5","5-5", 1-ppoint_srv2)
    set_trans("3-6","4-6", 1-ppoint_srv2)
    set_trans("6-4","SETv1", 1-ppoint_srv2)
    set_trans("5-5","6-5", 1-ppoint_srv2)
    set_trans("4-6","5-6", 1-ppoint_srv2)
    set_trans("3-6","4-6", 1-ppoint_srv2)
    set_trans("6-7","7-7", 1-ppoint_srv2)
    set_trans("7-6","SETv1", 1-ppoint_srv2)
    set_trans("7-7","6-5", 1-ppoint_srv2)

    # Absorbing states
    set_trans("SETv1","SETv1", 1)
    set_trans("SETv2","SETv2", 1)

    MC_tb = MarkovChain(tMat, STATES)
    return MC_tb

def resTIE(ppoint_srv1, ppoint_srv2, s_tb, graph=False):
    MC_tb = MCtb2(ppoint_srv1, ppoint_srv2)
    tMat = MC_tb.P
    s_tb = np.array(s_tb).reshape(1, -1)
    tMat_n = np.linalg.matrix_power(tMat, 1000)
    resTIE = np.dot(s_tb, tMat_n)
    if graph:
        # Build directed graph from transition matrix
        G = nx.DiGraph()
        states = MC_tb.state_values
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                prob = tMat[i, j]
                if prob > 0:
                    G.add_edge(from_state, to_state, weight=prob, label=f"{prob:.2f}")

        # Layout and draw
        plt.figure(figsize=(18, 12))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightgreen', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Markov Chain: Tie-break States")
        plt.axis('off')
        plt.show()
    return pd.DataFrame(resTIE, columns=MC_tb.state_values)

################## III - Set model #############################################
def MCset(phold1, phold2, ptie1):
    STATES = [
        "0-0","0-1","1-0","1-1",
        "2-0","0-2","3-0","2-1",
        "1-2","0-3","4-0","3-1",
        "2-2","1-3","0-4","5-0",
        "4-1", "3-2","2-3","1-4",
        "0-5","5-1","4-2","3-3",
        "2-4","1-5","5-2","4-3","3-4",
        "2-5","5-3","4-4","3-5","5-4",
        "4-5", "5-5","6-5","5-6",
        "6-6"
        ,"SETv1" # absorbing state
        ,"SETv2" # absorbing state
    ]
    idx = {state: i for i, state in enumerate(STATES)}
    tMat = np.zeros((len(STATES), len(STATES)))

    def set_trans(from_state, to_state, prob):
        tMat[idx[from_state], idx[to_state]] = prob

    # Player 1 serving
    set_trans("0-0","1-0", phold1)
    set_trans("2-0","3-0", phold1)
    set_trans("1-1","2-1", phold1)
    set_trans("0-2","1-2", phold1)
    set_trans("4-0","5-0", phold1)
    set_trans("3-1","4-1", phold1)
    set_trans("2-2","3-2", phold1)
    set_trans("1-3","2-3", phold1)
    set_trans("0-4","1-4", phold1)
    set_trans("5-1","SETv1", phold1)
    set_trans("4-2","5-2", phold1)
    set_trans("3-3","4-3", phold1)
    set_trans("2-4","3-4", phold1)
    set_trans("1-5","2-5", phold1)
    set_trans("5-3","SETv1", phold1)
    set_trans("4-4","5-4", phold1)
    set_trans("3-5","4-5", phold1)
    set_trans("5-5","6-5", phold1)

    set_trans("0-0","0-1", 1-phold1)
    set_trans("2-0","2-1", 1-phold1)
    set_trans("1-1","1-2", 1-phold1)
    set_trans("0-2","0-3", 1-phold1)
    set_trans("4-0","4-1", 1-phold1)
    set_trans("3-1","3-2", 1-phold1)
    set_trans("2-2","2-3", 1-phold1)
    set_trans("1-3","1-4", 1-phold1)
    set_trans("0-4","0-5", 1-phold1)
    set_trans("5-1","5-2", 1-phold1)
    set_trans("4-2","4-3", 1-phold1)
    set_trans("3-3","3-4", 1-phold1)
    set_trans("2-4","2-5", 1-phold1)
    set_trans("1-5","SETv2", 1-phold1)
    set_trans("5-3","5-4", 1-phold1)
    set_trans("4-4","4-5", 1-phold1)
    set_trans("3-5","SETv2", 1-phold1)
    set_trans("5-5","5-6", 1-phold1)

    # Player 2 serving
    set_trans("1-0","1-1", phold2)
    set_trans("0-1","0-2", phold2)
    set_trans("3-0","3-1", phold2)
    set_trans("2-1","2-2", phold2)
    set_trans("1-2","1-3", phold2)
    set_trans("0-3","0-4", phold2)
    set_trans("5-0","5-1", phold2)
    set_trans("4-1","4-2", phold2)
    set_trans("3-2","3-3", phold2)
    set_trans("2-3","2-4", phold2)
    set_trans("1-4","1-5", phold2)
    set_trans("0-5","SETv2", phold2)
    set_trans("5-2","5-3", phold2)
    set_trans("4-3","4-4", phold2)
    set_trans("3-4","3-5", phold2)
    set_trans("2-5","SETv2", phold2)
    set_trans("5-4","5-5", phold2)
    set_trans("4-5","SETv2", phold2)
    set_trans("5-6","SETv2", phold2)
    set_trans("6-5","6-6", phold2)

    set_trans("1-0","2-0", 1-phold2)
    set_trans("0-1","1-1", 1-phold2)
    set_trans("3-0","4-0", 1-phold2)
    set_trans("2-1","3-1", 1-phold2)
    set_trans("1-2","2-2", 1-phold2)
    set_trans("0-3","1-3", 1-phold2)
    set_trans("5-0","SETv1", 1-phold2)
    set_trans("4-1","5-1", 1-phold2)
    set_trans("3-2","4-2", 1-phold2)
    set_trans("2-3","3-3", 1-phold2)
    set_trans("1-4","2-4", 1-phold2)
    set_trans("0-5","1-5", 1-phold2)
    set_trans("5-2","SETv1", 1-phold2)
    set_trans("4-3","5-3", 1-phold2)
    set_trans("3-4","4-4", 1-phold2)
    set_trans("2-5","3-5", 1-phold2)
    set_trans("5-4","SETv1", 1-phold2)
    set_trans("4-5","5-5", 1-phold2)
    set_trans("5-6","6-6", 1-phold2)
    set_trans("6-5","SETv1", 1-phold2)

    # Absorbing states
    set_trans("SETv1","SETv1", 1)
    set_trans("SETv2","SETv2", 1)

    # Tie-break at 6-6
    set_trans("6-6","SETv1", ptie1)
    set_trans("6-6","SETv2", 1-ptie1)

    MC_set = MarkovChain(tMat, STATES)
    return MC_set

def resSET(phold1, phold2, ptie1, s_set, graph=False):
    MC_set = MCset(phold1, phold2, ptie1)
    tMat = MC_set.P
    s_set = np.array(s_set).reshape(1, -1)
    tMat_n = np.linalg.matrix_power(tMat, 100)
    resSET = np.dot(s_set, tMat_n)
    if graph:
        # Build directed graph from transition matrix
        G = nx.DiGraph()
        states = MC_set.state_values
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                prob = tMat[i, j]
                if prob > 0:
                    G.add_edge(from_state, to_state, weight=prob, label=f"{prob:.2f}")

        plt.figure(figsize=(18, 12))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightyellow', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
        plt.title("Markov Chain: Set States")
        plt.axis('off')
        plt.show()
    return pd.DataFrame(resSET, columns=MC_set.state_values)

################## IV - Match model ############################################
def MCmatch(pset_v1):
    pset_v2 = 1 - pset_v1
    STATES = ["0-0","0-1","1-0","1-1","2-0","0-2","2-1","1-2"
              ,"V1" # absorbing state
              ,"V2" # absorbing state
              ]
    idx = {state: i for i, state in enumerate(STATES)}
    tMat = np.zeros((10, 10))

    def set_trans(from_state, to_state, prob):
        tMat[idx[from_state], idx[to_state]] = prob

    # Set probabilities
    set_trans("0-0","1-0", pset_v1)
    set_trans("1-0","2-0", pset_v1)
    set_trans("0-1","1-1", pset_v1)
    set_trans("1-1","2-1", pset_v1)

    set_trans("0-0","0-1", pset_v2)
    set_trans("1-0","1-1", pset_v2)
    set_trans("0-1","0-2", pset_v2)
    set_trans("1-1","1-2", pset_v2)

    # Set stationary (absorbing) states
    set_trans("2-0", "V1", 1)
    set_trans("2-1", "V1", 1)
    set_trans("0-2", "V2", 1)
    set_trans("1-2", "V2", 1)
    set_trans("V1", "V1", 1)
    set_trans("V2", "V2", 1)

    MC_match = MarkovChain(tMat, STATES)
    return MC_match

def resMATCH(pset_v1, s_match, graph=False):
    MC_match = MCmatch(pset_v1)
    tMat = MC_match.P
    s_match = np.array(s_match).reshape(1, -1)
    tMat_n = np.linalg.matrix_power(tMat, 5)  # 2 sets, 5 steps is enough for absorption
    resMATCH = np.dot(s_match, tMat_n)
    if graph:
        G = nx.DiGraph()
        states = MC_match.state_values
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                prob = tMat[i, j]
                if prob > 0:
                    G.add_edge(from_state, to_state, weight=prob, label=f"{prob:.2f}")

        plt.figure(figsize=(12, 5))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightcoral', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkblue')
        plt.title("Markov Chain: Match States")
        plt.axis('off')
        plt.show()
    return pd.DataFrame(resMATCH, columns=MC_match.state_values)

############# V. Let's concatenate all of these blocks ##############
def predict1(gamescore, phold1, phold2, ptie1, pset_v1, s0match, s0set):
    """
    Computes match outcome probabilities when still in the first set.
    """
    s1match = s0match.copy()
    s1set = s0set.copy()

    # Set current game score in set state
    s1set.iloc[0, s1set.columns.get_loc("0-0")] = 0
    s1set.iloc[0, s1set.columns.get_loc(gamescore)] = 1

    # Compute probabilities for set completion from this score
    set_probs = resSET(phold1, phold2, ptie1, s1set)
    s1match.iloc[0, s1match.columns.get_loc("1-0")] = set_probs["SETv1"].iloc[0]
    s1match.iloc[0, s1match.columns.get_loc("0-1")] = set_probs["SETv2"].iloc[0]
    s1match.iloc[0, s1match.columns.get_loc("0-0")] = 0

    # Compute match outcome
    resTEST = resMATCH(pset_v1, s1match)
    return resTEST

def predict2(setscore, gamescore, phold1, phold2, ptie1, pset_v1, s0match, s0set):
    """
    Computes match outcome probabilities when in the second set.
    """
    s1match = s0match.copy()
    s1set = s0set.copy()

    # Use integer-based indexing with .iloc
    s1match.iloc[0, s1match.columns.get_loc("0-0")] = 0
    s1match.iloc[0, s1match.columns.get_loc(setscore)] = 1

    s1set.iloc[0, s1set.columns.get_loc("0-0")] = 0
    s1set.iloc[0, s1set.columns.get_loc(gamescore)] = 1

    set_probs = resSET(phold1, phold2, ptie1, s1set)

    if setscore == "1-0":
        s1match.iloc[0, s1match.columns.get_loc("2-0")] = set_probs["SETv1"].iloc[0]
        s1match.iloc[0, s1match.columns.get_loc("1-1")] = set_probs["SETv2"].iloc[0]
        s1match.iloc[0, s1match.columns.get_loc("1-0")] = 0
    elif setscore == "0-1":
        s1match.iloc[0, s1match.columns.get_loc("0-2")] = set_probs["SETv2"].iloc[0]
        s1match.iloc[0, s1match.columns.get_loc("1-1")] = set_probs["SETv1"].iloc[0]
        s1match.iloc[0, s1match.columns.get_loc("0-1")] = 0

    resTEST = resMATCH(pset_v1, s1match)
    return resTEST

def predict3(gamescore, phold1, phold2, ptie1, pset_v1, s0match, s0set):
    """
    Computes match outcome probabilities when in the third set.
    """
    s1match = s0match.copy()
    s1set = s0set.copy()
    setscore = "1-1"

    s1match.iloc[0, s1match.columns.get_loc("0-0")] = 0
    s1match.iloc[0, s1match.columns.get_loc(setscore)] = 1

    s1set.iloc[0, s1set.columns.get_loc("0-0")] = 0
    s1set.iloc[0, s1set.columns.get_loc(gamescore)] = 1

    set_probs = resSET(phold1, phold2, ptie1, s1set)
    s1match.iloc[0, s1match.columns.get_loc("2-1")] = set_probs["SETv1"].iloc[0]
    s1match.iloc[0, s1match.columns.get_loc("1-2")] = set_probs["SETv2"].iloc[0]
    s1match.iloc[0, s1match.columns.get_loc("1-1")] = 0

    resTEST = resMATCH(pset_v1, s1match)
    return resTEST

def determiMM(ppoint_srv1, ppoint_srv2, setscore, gamescore, s0match, s0set, s0game, s0tb):
    """
    Computes match outcome probabilities, given the score and point/game/set probabilities.
    """
    # Compute intermediate probabilities
    phold1 = resGAME(ppoint_srv1, s0game)["HOLD"].iloc[0]
    phold2 = resGAME(ppoint_srv2, s0game)["HOLD"].iloc[0]
    ptie1 = resTIE(ppoint_srv1, ppoint_srv2, s0tb)["SETv1"].iloc[0]
    pset_v1 = resSET(phold1, phold2, ptie1, s0set)["SETv1"].iloc[0]

    print(f"Let's modelize this match from the score: {setscore} Sets, {gamescore} Games")

    if setscore == "0-0":
        resTEST = predict1(gamescore, phold1, phold2, ptie1, pset_v1, s0match, s0set)
    elif setscore == "1-1":
        resTEST = predict3(gamescore, phold1, phold2, ptie1, pset_v1, s0match, s0set)
    elif setscore in ["1-0", "0-1"]:
        resTEST = predict2(setscore, gamescore, phold1, phold2, ptie1, pset_v1, s0match, s0set)
    else:
        raise ValueError("Invalid setscore provided.")

    return resTEST
