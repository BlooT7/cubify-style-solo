# cubify-style-solo
 The solo version of cubify style implementation. The whole program is finished by myself. I borrowed my own work from the mesh project as a stencil in the class.
# How to use the code?
 I use Qt Creator as the implementation IDE, so I only test the program through the IDE. The build version I used is Qt 5.12.10 MinGW 64-bit. I left the shadow building off to ensure it could find the right directory. The running argument format is "INPUT_FILE OUTPUT_FILE MODE LAMBDA REMOVING_NUMBER_EDGES OUTPUT_LOSS_FLAG" where "MODE" could be "regular" or "fast" which corresponds to fast version of stylization in the paper. "INPUT_FILE" and "OUTPUT_FILE" are just directories and names of input and output files. "LAMBDA" is the only control variable for stylization mentioned in the paper: a higher lambda variable means more cubeness and vice versa. "REMOVING_NUMBER_EDGES" is functioning literally, which is only used for the fast algorithm. This argument will be omitted for regular version of the algorithm. The last flag is used for control if we want to print the loss during running. 
# Data
 input algorithm lambda removing-edges iteration time
 bunny regular 
 bunny fast 0.3 5000
