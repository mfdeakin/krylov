
#ifndef _CONSTANTS_HPP_
#define _CONSTANTS_HPP_

#include <limits>

#include "xtensor/xtensor.hpp"

using real = double;

using triple = std::tuple<real, real, real>;

using matrix = xt::xtensor<real, 2>;
using vector = xt::xtensor<real, 1>;

constexpr real pi = 3.1415926535897932384626433832795;
constexpr real q_nan = std::numeric_limits<real>::quiet_NaN();
constexpr real s_nan = std::numeric_limits<real>::signaling_NaN();

#endif // _CONSTANTS_HPP_
