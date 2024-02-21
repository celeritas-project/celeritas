//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalDistributionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Pre- and post-step data for sampling optical photons.
 */
struct OpticalStepData
{
    units::LightSpeed speed;
    Real3 pos{};

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return speed > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input data for sampling optical photons.
 */
struct OpticalDistributionData
{
    size_type num_photons{};  //!< Sampled number of photons to generate
    real_type time{};  //!< Pre-step time
    real_type step_length{};
    units::ElementaryCharge charge;
    OpticalMaterialId material;
    EnumArray<StepPoint, OpticalStepData> points;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_photons > 0 && step_length > 0 && material;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
