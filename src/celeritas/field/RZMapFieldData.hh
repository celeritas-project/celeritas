//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * FieldMap (2-dimensional RZ map) parameters
 */
struct FieldMapParameters
{
    size_type num_grid_r;
    size_type num_grid_z;
    real_type delta_grid;
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
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData
{
    //! Parameters of FieldMap
    FieldMapParameters params;

    //! Options for FieldDriver
    FieldDriverOptions options;

    //! Index of FieldMap Collection
    using ElementId = ItemId<size_type>;

    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    ElementItems<FieldMapElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    inline CELER_FUNCTION bool valid(size_type idx_z, size_type idx_r) const
    {
        return (idx_z < params.num_grid_z && idx_r < params.num_grid_r);
    }

    inline CELER_FUNCTION ElementId id(int idx_z, int idx_r) const
    {
        return ElementId(idx_z * params.num_grid_r + idx_r);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RZMapFieldParamsData& operator=(RZMapFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        params = other.params;
        options = other.options;
        fieldmap = other.fieldmap;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
