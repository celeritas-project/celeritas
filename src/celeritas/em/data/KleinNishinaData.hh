//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/KleinNishinaData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Model and particles IDs.
 */
struct KleinNishinaIds
{
    ParticleId electron;
    ParticleId gamma;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return electron && gamma; }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a KleinNishinaInteractor.
 */
struct KleinNishinaData
{
    using Mass = units::MevMass;

    //! Model and particle identifiers
    KleinNishinaIds ids;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && inv_electron_mass > 0;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
