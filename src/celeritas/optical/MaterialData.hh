//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Material properties used to calculate Rayleigh mean free paths.
 */
struct OpticalRayleighMaterial
{
    real_type scale_factor{1};  //!< User scale for the scattering length
    real_type compressibility{0};  //!< Isothermal compressibility
    real_type temperature{0};  //!< Material temperature

    operator bool() const
    {
        return scale_factor > 0 && compressibility > 0 && temperature > 0;
    }
};

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

    OpticalMaterialItems<GenericGridRecord> refractive_index;
    VolumeItems<OpticalMaterialId> optical_id;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !refractive_index.empty() && !optical_id.empty()
               && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    MaterialParamsData& operator=(MaterialParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        refractive_index = other.refractive_index;
        optical_id = other.optical_id;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
