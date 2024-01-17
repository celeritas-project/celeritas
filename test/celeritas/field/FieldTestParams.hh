//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldTestParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

struct FieldTestParams
{
    size_type nstates{};  //! number of states (tracks)
    int nsteps{};  //! number of steps/revolution
    int revolutions{};  //! number of revolutions
    real_type field_value{};  //! field value along z [tesla]
    real_type radius{};  //! radius of curvature [cm]
    real_type delta_z{};  //! z-change/revolution [cm]
    real_type energy{};  //! energy of the test particle
    real_type momentum_y{};  //! initial momentum_y [MeV/c]
    real_type momentum_z{};  //! initial momentum_z [MeV/c]
    real_type epsilon{};  //! tolerance error
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
