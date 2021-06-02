//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldMapInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/Collection.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * FieldMap (2-dimensional RZ map) parameters
 */
struct FieldMapParameters
{
    size_type num_grid_r;
    size_type num_grid_z;
    real_type offset_z;
};

//---------------------------------------------------------------------------//
/*!
 * FieldMap element
 */
struct FieldMapElement
{
    float value_z;
    float value_r;
};

//---------------------------------------------------------------------------//
/*!
 * FieldMap data: vector of size [num_grid_z*num_grid_r] which stores data for
 * the equivalent 2-dimensional RZ-array[num_grid_z][num_grid_r] and
 * associated parameters
 */
struct FieldMapData
{
    FieldMapParameters           params;
    std::vector<FieldMapElement> data;
};

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct FieldMapGroup
{
    //! Parameters of FieldMap
    FieldMapParameters params;

    //! Index of FieldMap Collection
    using ElementId = celeritas::ItemId<size_type>;

    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;
    ElementItems<FieldMapElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    inline CELER_FUNCTION bool valid(size_type iz, size_type ir) const
    {
        return (iz < params.num_grid_z && ir < params.num_grid_r);
    }

    inline CELER_FUNCTION ItemId<size_type> id(int iz, int ir) const
    {
        return ElementId(iz * params.num_grid_r + ir);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    FieldMapGroup& operator=(const FieldMapGroup<W2, M2>& other)
    {
        CELER_EXPECT(other);
        params   = other.params;
        fieldmap = other.fieldmap;
        return *this;
    }
};

using FieldMapDeviceRef
    = FieldMapGroup<Ownership::const_reference, MemSpace::device>;
using FieldMapHostRef
    = FieldMapGroup<Ownership::const_reference, MemSpace::host>;
using FieldMapNativeRef
    = FieldMapGroup<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
