Copyright 1995 Geometry Center, University of Minnesota.

This directory contains the C++ program "evert" for turning the sphere
inside out without tears, creases, or pinches. The program was written
by Nathaniel Thurston, and is based on ideas of Bill Thurston.
Silvio Levy wrote the documentation and organized the distribution.

This software was developed at the Geometry Center (University of Minnesota).

The documentation consists of a Latex file manual.tex, also available
in Postscript (manual.ps).  It assumes you are familiar with the basic
ideas of the algorithm for turning the sphere inside out described in
the computer-animated video Outside In.  This video, directed by
Silvio Levy, Tamara Munzner and Delle Maxwell, gives an elementary
description of the eversion, and shows it step by step, from different
points of view. You should also consult the full-color illustrated
book Making Waves: A Guide to the Ideas Behind Outside In, which comes
with the video.  Outside In and Making Waves are distributed by

     A K Peters
     289 Linden Street
     Wellesley MA 02181
     phone 617-235-2210
     fax 617-235-2404

You can also point your favorite Web browser to http://www.geom.uiuc.edu/locate/oi.

PREPARATION

Evert is written in C++, GNU dialect (gcc) as current circa 1995.
If you have more recent information, a pull request updating this
paragraph would be welcome.

Evert takes no input, other than command-line options. Its output
is a file of numbers representing the sphere (or, more usually, a
piece of it) at a given point in the eversion. This means that, in
order to see the output, you need to be able to transform the numbers
-- that is, the geometry of the sphere into a picture. There are
various programs that do this. One is called Geomview, and is freely
available from geomview.org.

If you don't have Geomview but have a different viewing program,
you may be able to adapt the output of evert to serve as input to your
program, either by changing the code or by writing an output filter.
If you do that, we would appreciate hearing from you and getting a
copy of your code.

For information on how to run evert, see manual.tex or manual.ps.
