//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/SafetyCacheData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Initializer sets a flag
struct SafetyCacheInitializer
{
    bool use_safety{true};
};

//---------------------------------------------------------------------------//
/*!
 * Store and access the safety cache.
 */
template<Ownership W, MemSpace M>
struct SafetyCacheStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<real_type> radius;
    Items<Real3> origin;

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !radius.empty() && origin.size() == radius.size();
    }

    //! State size
    CELER_FUNCTION ThreadId::size_type size() const { return radius.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SafetyCacheStateData& operator=(SafetyCacheStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        radius = other.radius;
        origin = other.origin;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize safety cache states.
 */
template<MemSpace M>
void resize(SafetyCacheStateData<Ownership::value, M>* data, size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&data->radius, size);
    resize(&data->origin, size);
    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
