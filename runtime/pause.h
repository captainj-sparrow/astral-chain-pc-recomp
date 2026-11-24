// The game's pause (pause.c).
#pragma once
#include "nvn_obj.h"

// At the end of a present, on the thread that presented t, with its virtual core given up: returns at once unless the
// game has to pause, else once the pause is over.
void pause_frame(const tex_t *t);
