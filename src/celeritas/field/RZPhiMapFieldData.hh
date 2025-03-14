//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/math/Turn.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * FieldMap (3-dimensional RZ-Phi map) grid data
 */
template<Ownership W, MemSpace M>
struct RZPhiMapGridData
{
    template<class T>
    using Items = Collection<T, W, M>;
    Items<real_type> storage;  //!< [Phi, R, Z]
    Array<size_type, 3> grid_size;  //!< [Phi, R, Z]

    ItemRange<real_type> phi;  //!< Index range for phi
    ItemRange<real_type> r;  //!< Index range for r
    ItemRange<real_type> z;  //!< Index range for z

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !storage.empty() && grid_size[0] > 1 && grid_size[1] > 1
               && grid_size[2] > 1 && phi.size() == grid_size[0]
               && r.size() == grid_size[1] && z.size() == grid_size[2];
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RZPhiMapGridData& operator=(RZPhiMapGridData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        storage = other.storage;
        grid_size = other.grid_size;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * FieldMap element for RZ-Phi map
 */
struct RZPhiMapElement
{
    real_type value_z;
    real_type value_r;
    real_type value_phi;
};

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct RZPhiMapFieldParamsData
{
    //! Grids of FieldMap
    RZPhiMapGridData<W, M> grids;

    //! Options for FieldDriver
    FieldDriverOptions options;

    //! Index of FieldMap Collection
    using ElementId = ItemId<size_type>;

    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //! FieldMap data
    ElementItems<RZPhiMapElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    //! Check if the given position is within the field map bounds
    inline CELER_FUNCTION bool valid(real_type z, real_type r, Turn phi) const
    {
        CELER_EXPECT(grids);
        return (z >= grids.storage[grids.z.front()]
                && z <= grids.storage[grids.z.back()]
                && r >= grids.storage[grids.r.front()]
                && r <= grids.storage[grids.r.back()]
                && phi.value() >= grids.storage[grids.phi.front()]
                && phi.value() <= grids.storage[grids.phi.back()]);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RZPhiMapFieldParamsData&
    operator=(RZPhiMapFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        grids = other.grids;
        options = other.options;
        fieldmap = other.fieldmap;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas