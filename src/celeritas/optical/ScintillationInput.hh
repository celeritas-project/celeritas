//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input data for sampling optical photons for a given scintillation step.
 */
struct ScintillationInput
{
    size_type num_photons{};  //!< Number of photons to generate
    real_type step_length{};  //!< Step length
    real_type time{};  //!< Pre-step time
    real_type pre_velocity{};  //!< Pre-step velocity
    real_type post_velocity{};  //!< Post-step velocity
    Real3 pre_pos{};  //!< Pre-step position
    Real3 post_pos{};  //!< Post-step position
    units::ElementaryCharge charge{};  //!< Particle charge
    OpticalMaterialId matId{};  // !< OpticalMaterial Id

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_photons > 0 && step_length > 0 && time >= 0 && matId;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
