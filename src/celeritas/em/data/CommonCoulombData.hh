//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
//! Opaque index to MSC-applicable particles
using MscParticleId = OpaqueId<struct MscParticle_>;

//---------------------------------------------------------------------------//
/*!
 * Physics IDs for MSC.
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

}  // namespace celeritas
