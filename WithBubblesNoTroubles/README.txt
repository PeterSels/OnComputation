WithBubblesNoTroubles.com
positions as many bubbles as possible in 2D space

given data:
persons p \in P
bubbles b \in B
each person p is an element of just one bubble

constraints:
The distance between each pair of persons of a different bubbles is to be kept larger than a minimum distance Mpush.
In covid Mpush = 1.5m.
The distance between each pair of person of the smae bubble is to be kept smaller than a minimum distance Mpull.
This is user setting decidable. e.g. Mpull=5m.

grid:
a grid of possible (x,y) sets are specifyable.
no grid:
If none is specified, placement can be off grid.
limited or not by
xlo <= x < xhi
ylo <= y < yhi

objective function:
maximise 
sum of distances between all people belonging to different bubbles.
-
sum of distances between all people belonging to the same bubble.

supergrid:
If a grid exists, collections of grid point can be defined as 'tables' so that bubbles should be entirely mapped on tables.

