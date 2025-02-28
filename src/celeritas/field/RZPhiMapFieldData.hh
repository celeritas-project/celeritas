//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/UniformGridData.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * FieldMap (3-dimensional RZ-Phi map) grid data
 */
struct RZPhiMapGridData
{
    UniformGridData data_z;
    UniformGridData data_r;
    UniformGridData data_phi;
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
    RZPhiMapGridData grids;

    //! Options for FieldDriver
    FieldDriverOptions options;

    //! Index of FieldMap Collection
    using ElementId = ItemId<size_type>;

    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    ElementItems<RZPhiMapElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    inline CELER_FUNCTION bool
    valid(real_type z, real_type r, real_type phi) const
    {
        CELER_EXPECT(grids.data_z);
        CELER_EXPECT(grids.data_r);
        CELER_EXPECT(grids.data_phi);
        return (z >= grids.data_z.front && z <= grids.data_z.back
                && r >= grids.data_r.front && r <= grids.data_r.back
                && phi >= grids.data_phi.front && phi <= grids.data_phi.back);
    }

    inline CELER_FUNCTION ElementId id(size_type idx_z,
                                       size_type idx_r,
                                       size_type idx_phi) const
    {
        CELER_EXPECT(grids.data_r);
        CELER_EXPECT(grids.data_phi);
        // Index with ordering [Z][R][Phi]
        return ElementId(idx_z * grids.data_r.size * grids.data_phi.size
                         + idx_r * grids.data_phi.size + idx_phi);
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