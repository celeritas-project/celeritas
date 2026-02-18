//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
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
    using Grid = NonuniformGridRecord;
    using GridId = OpaqueId<Grid>;

    template<class T>
    using Items = Collection<T, W, M>;

    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;

    using ReflectivityGrids = EnumArray<ReflectivityAction, SurfaceItems<Grid>>;
    //!@}

    //// DATA ////

    //! Reflectivity and transmittance grids
    ReflectivityGrids reflectivity;

    //! Optional quantum efficiency grids
    SurfaceItems<GridId> efficiency_ids;
    Items<Grid> efficiency;

    //! Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !reflectivity[ReflectivityAction::interact].empty()
               && !reflectivity[ReflectivityAction::transmit].empty()
               && !efficiency_ids.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GridReflectivityData<W, M>&
    operator=(GridReflectivityData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        for (auto action : range(ReflectivityAction::size_))
        {
            reflectivity[action] = other.reflectivity[action];
        }
        efficiency_ids = other.efficiency_ids;
        efficiency = other.efficiency;
        reals = other.reals;

        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
