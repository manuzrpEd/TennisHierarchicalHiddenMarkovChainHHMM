Can we characterize a Tennis match as a Hidden Markov Model?

This repo helps modelize a tennis match with the help of markov chains. \
Because of the particular structure of tennis scores, i.e. imbricated sequential models which 
are perfectly described with Hierarchical Hidden Markov Models (HHMM). \
The match model is "hidden" because:\
    - We don't observe why a point was won (e.g., unforced error, ace, fatigue).\
    - The observed outcome (win/loss of point) is a result of latent variables (skill, momentum, psychological state).\
Given our only two inputs - the probabilities of each player winning a point on his serve we can modelize a game and a tie-break first, then we can use these modelizations to modelize a set, and finally we can use the set model to modelize a whole match.
