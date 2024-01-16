//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CerenkovData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Cerenkov angle integrals tablulated as a function of photon energy.
 */
template<Ownership W, MemSpace M>
struct CerenkovData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    Items<real_type> reals;
    OpticalMaterialItems<GenericGridData> angle_integral;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !angle_integral.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CerenkovData& operator=(CerenkovData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        angle_integral = other.angle_integral;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
