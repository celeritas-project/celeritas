//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGeneratorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Data for sampling optical photons from user-configurable distributions.
 */
struct PrimaryDistributionData
{
    size_type num_photons{};
    OnedDistributionId energy;
    ThreedDistributionId angle;
    ThreedDistributionId shape;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_photons > 0 && energy && angle && shape;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
