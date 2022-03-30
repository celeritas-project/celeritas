//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a KleinNishinaInteractor.
 */
struct KleinNishinaData
{
    using Mass = units::MevMass;

    struct
    {
        //! Model ID
        ModelId model;
        //! ID of an electron
        ParticleId electron;
        //! ID of a gamma
        ParticleId gamma;
    } ids;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids.model && ids.electron && ids.gamma && inv_electron_mass > 0;
    }
};

using KleinNishinaDeviceRef = KleinNishinaData;
using KleinNishinaHostRef   = KleinNishinaData;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
