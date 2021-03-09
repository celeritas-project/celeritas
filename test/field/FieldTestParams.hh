//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTestParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

struct FieldTestParams
{
    size_type nstates;
    int       nsteps;
    int       revolutions;
    real_type field_value;
    real_type radius;
    real_type delta_z;
    real_type energy;
    real_type momentum_y;
    real_type momentum_z;
    real_type epsilon;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
