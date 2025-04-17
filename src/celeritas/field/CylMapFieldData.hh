//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/math/Turn.hh"
#include "celeritas/Types.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//! Real type for cylindrical map field data
using cylmap_real_type = float;

//---------------------------------------------------------------------------//
/*!
 * MapField (3-dimensional R-Phi-Z map) grid data
 */
template<Ownership W, MemSpace M>
struct CylMapGridData
{
    using real_type = cylmap_real_type;

    template<class T>
    using Items = Collection<T, W, M>;
    Items<real_type> storage;  //!< [R, Phi, Z]
    EnumArray<CylAxis, ItemRange<real_type>> axes;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !storage.empty()
               && storage.size()
                      == axes[CylAxis::r].size() + axes[CylAxis::phi].size()
                             + axes[CylAxis::z].size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CylMapGridData& operator=(CylMapGridData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        storage = other.storage;
        axes = other.axes;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct CylMapFieldParamsData
{
    using real_type = cylmap_real_type;

    //! Grids of MapField
    CylMapGridData<W, M> grids;

    //! Field propagation and substepping tolerances
    FieldDriverOptions options;

    //! Index of MapField Collection
    using ElementId = ItemId<size_type>;

    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //! MapField data
    ElementItems<EnumArray<CylAxis, real_type>> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    //! Check if the given position is within the field map bounds
    inline CELER_FUNCTION bool
    valid(real_type r, Turn_t<real_type> phi, real_type z) const
    {
        CELER_EXPECT(grids);
        return (
            r >= grids.storage[grids.axes[CylAxis::r].front()]
            && r <= grids.storage[grids.axes[CylAxis::r].back()]
            && phi.value() >= grids.storage[grids.axes[CylAxis::phi].front()]
            && phi.value() <= grids.storage[grids.axes[CylAxis::phi].back()]
            && z >= grids.storage[grids.axes[CylAxis::z].front()]
            && z <= grids.storage[grids.axes[CylAxis::z].back()]);
    }

    inline CELER_FUNCTION ElementId id(size_type idx_r,
                                       size_type idx_phi,
                                       size_type idx_z) const
    {
        CELER_EXPECT(grids);
        Array<size_type, static_cast<size_type>(CylAxis::size_)> tmp{
            grids.axes[CylAxis::r].size(),
            grids.axes[CylAxis::phi].size(),
            grids.axes[CylAxis::z].size()};
        return ElementId{HyperslabIndexer{tmp}(idx_r, idx_phi, idx_z)};
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CylMapFieldParamsData& operator=(CylMapFieldParamsData<W2, M2> const& other)
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