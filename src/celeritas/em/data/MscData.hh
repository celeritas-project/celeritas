//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MscData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Common settable parameters and default values for multiple scattering.
 */
struct MscParameters
{
    real_type lambda_limit{1 * units::millimeter};  //!< lambda limit
    real_type geom_fact{2.5};  //!< geometry factor
    real_type range_fact{0.04};  //!< range factor for e-/e+ (0.2 for muon/h)
    real_type safety_fact{0.6};  //!< safety factor
};

//---------------------------------------------------------------------------//
/*!
 * Physics IDs for MSC.
 *
 * TODO these will probably be changed to a map over all particle IDs.
 */
struct MscIds
{
    ParticleId electron;
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron && positron;
    }
};

}  // namespace celeritas
