//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/CommonCoulombData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics IDs for MSC.
 *
 * TODO these will probably be changed to a map over all particle IDs.
 */
struct CoulombIds
{
    ParticleId electron;
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron && positron;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Parameters used in both single and multiple Coulomb scattering models.
 */
struct CoulombParameters
{
    //! Whether to use combined single and multiple scattering
    bool is_combined;
    //! Polar angle limit between single and multiple scattering
    real_type costheta_limit;
    //! Factor for the screening coefficient
    real_type screening_factor;
    //! Factor used to calculate the maximum scattering angle off of a nucleus
    real_type a_sq_factor;

    explicit CELER_FUNCTION operator bool() const
    {
        return costheta_limit >= -1 && costheta_limit <= 1
               && screening_factor > 0 && a_sq_factor > 0;
    }
};

}  // namespace celeritas
