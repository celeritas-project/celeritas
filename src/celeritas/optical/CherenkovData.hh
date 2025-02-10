//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CherenkovData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Cherenkov angle integrals tablulated as a function of photon energy.
 */
template<Ownership W, MemSpace M>
struct CherenkovData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    OpticalMaterialItems<NonuniformGridRecord> angle_integral;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !angle_integral.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CherenkovData& operator=(CherenkovData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        angle_integral = other.angle_integral;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
