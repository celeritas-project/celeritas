//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_COVFIE || __DOXYGEN__

#    include "corecel/Macros.hh"
#    include "corecel/Types.hh"

#    include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Primary template; specializations for value/host, const_reference/host, and
// value/device are in RZMapFieldData.covfie.hh (requires C++20 and covfie).
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData;

// const_reference/device is defined here (no covfie types needed) so that
// public headers included by C++17 TUs can use
// NativeCRef<RZMapFieldParamsData> on device without pulling in covfie's C++20
// headers.
template<>
struct RZMapFieldParamsData<Ownership::const_reference, MemSpace::device>
{
    CELER_FUNCTION explicit operator bool() const { return field_view; }

    FieldDriverOptions options;
    //! Opaque pointer to view_t in device memory; cast to view_t const* in .cu
    void const* field_view{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#else

#    include "corecel/Macros.hh"
#    include "corecel/Types.hh"
#    include "corecel/data/Collection.hh"
#    include "corecel/grid/UniformGridData.hh"

#    include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * MapField (2-dimensional RZ map) grid data
 */
struct MapFieldGridData
{
    UniformGridData data_z;
    UniformGridData data_r;
};

//---------------------------------------------------------------------------//
/*!
 * MapField element
 */
struct MapFieldElement
{
    real_type value_z;
    real_type value_r;
};

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData
{
    //! Grids of MapField
    MapFieldGridData grids;

    //! Field propagation and substepping tolerances
    FieldDriverOptions options;

    //! Index of MapField Collection
    using ElementId = ItemId<size_type>;

    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    ElementItems<MapFieldElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    inline CELER_FUNCTION bool valid(real_type z, real_type r) const
    {
        CELER_EXPECT(grids.data_z);
        CELER_EXPECT(grids.data_r);
        return (z >= grids.data_z.front && z <= grids.data_z.back
                && r >= grids.data_r.front && r <= grids.data_r.back);
    }

    inline CELER_FUNCTION ElementId id(size_type idx_z, size_type idx_r) const
    {
        CELER_EXPECT(grids.data_r);
        return ElementId(idx_z * grids.data_r.size + idx_r);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RZMapFieldParamsData& operator=(RZMapFieldParamsData<W2, M2> const& other)
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
#endif  // CELERITAS_USE_COVFIE
