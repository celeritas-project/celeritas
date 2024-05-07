//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/CoulombScatteringData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "CommonCoulombData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Constant shared data used by the CoulombScatteringModel.
 */
struct CoulombScatteringData
{
    // Particle IDs
    CoulombIds ids;

    // Action ID
    ActionId action;

    //! Maximum scattering polar angle
    static CELER_CONSTEXPR_FUNCTION real_type costheta_max() { return -1; }

    // Check if the data is initialized
    explicit CELER_FUNCTION operator bool() const { return ids && action; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
