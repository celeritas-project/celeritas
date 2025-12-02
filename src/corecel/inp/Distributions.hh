//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/inp/Distributions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//! "Sample" the given value
template<class T>
struct DeltaDistribution
{
    T value{};
};

//---------------------------------------------------------------------------//
//! Sample from a Gaussian (normal) distribution
struct NormalDistribution
{
    double mean{0};
    double stddev{1};
};

//---------------------------------------------------------------------------//
//! Sample a point uniformly on the surface of a unit sphere
struct IsotropicDistribution
{
};

//---------------------------------------------------------------------------//
//! Sample uniformly in a box
struct UniformBoxDistribution
{
    using Real3 = Array<double, 3>;

    Real3 lower{0, 0, 0};
    Real3 upper{0, 0, 0};
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
