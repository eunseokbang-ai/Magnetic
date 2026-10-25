"""magcore - the magnetic-survey processing library shared by DroneMag
Studio (backend/app) and the landmine-detection program.

Every module here takes arrays and returns arrays or small result
objects; nothing in it knows about HTTP, projects or the frontend. The
one piece of environment it needs - where to write caches - is given by
paths.configure().
"""
