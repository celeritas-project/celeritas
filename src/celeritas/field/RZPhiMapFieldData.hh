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
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/math/Turn.hh"
#include "celeritas/Types.hh"

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
    EnumArray<CylAxis, size_type> grid_size;
    EnumArray<CylAxis, ItemRange<real_type>> axes;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !storage.empty() && grid_size[CylAxis::Phi] > 1
               && grid_size[CylAxis::R] > 1 && grid_size[CylAxis::Z] > 1
               && axes[CylAxis::Phi].size() == grid_size[CylAxis::Phi]
               && axes[CylAxis::R].size() == grid_size[CylAxis::R]
               && axes[CylAxis::Z].size() == grid_size[CylAxis::Z];
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RZPhiMapGridData& operator=(RZPhiMapGridData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        storage = other.storage;
        grid_size = other.grid_size;
        axes = other.axes;
        return *this;
    }
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
    ElementItems<EnumArray<CylAxis, real_type>> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    //! Check if the given position is within the field map bounds
    inline CELER_FUNCTION bool valid(real_type z, real_type r, Turn phi) const
    {
        CELER_EXPECT(grids);
        return (
            z >= grids.storage[grids.axes[CylAxis::Z].front()]
            && z <= grids.storage[grids.axes[CylAxis::Z].back()]
            && r >= grids.storage[grids.axes[CylAxis::R].front()]
            && r <= grids.storage[grids.axes[CylAxis::R].back()]
            && phi.value() >= grids.storage[grids.axes[CylAxis::Phi].front()]
            && phi.value() <= grids.storage[grids.axes[CylAxis::Phi].back()]);
    }

    inline CELER_FUNCTION ElementId id(size_type idx_phi,
                                       size_type idx_r,
                                       size_type idx_z) const
    {
        CELER_EXPECT(grids);
        // HyperSlabIndexer does not take Array<T const>
        Array<size_type, static_cast<size_type>(CylAxis::size_)> tmp{
            grids.grid_size[CylAxis::Phi],
            grids.grid_size[CylAxis::R],
            grids.grid_size[CylAxis::Z]};
        return ElementId{HyperslabIndexer{tmp}(idx_phi, idx_r, idx_z)};
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