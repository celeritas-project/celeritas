//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/OpticalRayleighData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Per optical material data used for OpticalRayleigh scattering.
 */
template<Ownership W, MemSpace M>
struct OpticalRayleighData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;
    //!@}

    OpticalMaterialItems<real_type> scale_factor;
    OpticalMaterialItems<real_type> compressibility;

    //! True if data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !scale_factor.empty() && !compressibility.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalRayleighData& operator=(OpticalRayleighData<W2, M2> const& other)
    {
        scale_factor = other.scale_factor;
        compressibility = other.compressibility;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
