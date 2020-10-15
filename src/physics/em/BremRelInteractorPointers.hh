//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremRelInteractorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Elemental data
 */
struct BremElement
{
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct BremRelInteractorPointers
{
    //! ID of an electron
    ParticleDefId electron_id;
    //! ID of a gamma
    ParticleDefId gamma_id;
    //! Whether to correct for the Landau-Pomeranchuk-Migdal effect
    bool use_lpm;
    //! Migdal's constant: 4 * pi * r_e * l_ec^2
    real_type migdal_constant;
    //! LPM constant: alpha * (m_e c^2)^2 / (4 * pi * hbar * c)
    real_type lpm_constant; // [MeV/cm]

    //! Elemental data needed for sampling [access by ElementId]
    span<const BremElement> elements;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_id && gamma_id; // XXX
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
