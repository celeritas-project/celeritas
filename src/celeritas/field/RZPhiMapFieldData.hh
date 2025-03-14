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
    Items<real_type> z;
    Items<real_type> r;
    Items<Turn> phi;
    Array<size_type, 3> grid_size;  //!< [Z, R, Phi]

    explicit inline CELER_FUNCTION operator bool() const
    {
        return !z.empty() && !r.empty() && !phi.empty();
    }
    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RZPhiMapGridData& operator=(RZPhiMapGridData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        z = other.z;
        r = other.r;
        phi = other.phi;
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
    ElementItems<RZPhiMapElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    inline CELER_FUNCTION bool
    valid(real_type z, real_type r, real_type phi) const
    {
        CELER_EXPECT(grids);
        Turn turn_phi{phi / Turn::unit_type::value()};
        auto view_z
            = Span<real_type const>{grids.z.data().get(), grids.z.size()};
        auto view_r
            = Span<real_type const>{grids.r.data().get(), grids.r.size()};
        auto view_phi
            = Span<Turn const>{grids.phi.data().get(), grids.phi.size()};
        return (z >= view_z.front() && z <= view_z.back()
                && r >= view_r.front() && r <= view_r.back()
                && turn_phi >= view_phi.front() && turn_phi <= view_phi.back());
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