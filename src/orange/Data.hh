//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Data.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/OpaqueId.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Data for type-deleted surface definitions.
 *
 * Surfaces each have a compile-time number of real data needed to define them.
 * (These usually are the nonzero coefficients of the quadric equation.) A
 * surface ID points to an offset into the `data` field. These surface IDs are
 * *global* over all universes.
 */
template<Ownership W, MemSpace M>
struct SurfaceData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M, SurfaceId>;

    //// DATA ////

    Items<SurfaceType>          types;
    Items<OpaqueId<real_type>>  offsets;
    Collection<real_type, W, M> reals;

    //// METHODS ////

    //! Number of surfaces
    CELER_FUNCTION SurfaceId::size_type size() const { return types.size(); }

    //! True if sizes are valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !types.empty() && offsets.size() == types.size()
               && reals.size() >= types.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfaceData& operator=(const SurfaceData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        types   = other.types;
        offsets = other.offsets;
        reals   = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
