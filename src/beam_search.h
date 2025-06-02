#pragma once

#include <vector>
#include <string>

class Instance;
class MLP;

double BS(double time_limit, int beam_width, Instance* inst, MLP& neural_network, bool training);

