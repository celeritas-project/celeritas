//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Storage for grid reflectivity data.
 */
template<Ownership W, MemSpace M>
struct GridReflectivityData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;

    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    //// DATA ////

    //! Surface reflectivity data
    SurfaceItems<NonuniformGridRecord> reflectivity;

    //! Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !reflectivity.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GridReflectivityData<W, M>&
    operator=(GridReflectivityData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        reflectivity = other.reflectivity;
        reals = other.reals;

        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
