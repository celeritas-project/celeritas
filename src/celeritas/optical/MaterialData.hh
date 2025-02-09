//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridData.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Shared optical material properties.
 */
template<Ownership W, MemSpace M>
struct MaterialParamsData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;
    template<class T>
    using VolumeItems = celeritas::Collection<T, W, M, VolumeId>;

    //// MEMBER DATA ////

    OpticalMaterialItems<NonuniformGridRecord> refractive_index;
    VolumeItems<OpticalMaterialId> optical_id;
    OpticalMaterialItems<CoreMaterialId> core_material_id;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !refractive_index.empty() && !optical_id.empty()
               && !core_material_id.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    MaterialParamsData& operator=(MaterialParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        refractive_index = other.refractive_index;
        optical_id = other.optical_id;
        core_material_id = other.core_material_id;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
