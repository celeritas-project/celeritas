//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Material dependent scintillation property.
 */
struct ScintillationComponent
{
    real_type yield_prob{};  //!< Probability of the yield
    real_type lambda_mean{};  //!< Mean wavelength
    real_type lambda_sigma{};  //!< Standard dev. of wavelength
    real_type rise_time{};  //!< Rise time
    real_type fall_time{};  //!< Decay time

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield_prob > 0 && lambda_mean > 0 && lambda_sigma > 0
               && rise_time >= 0 && fall_time > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data characterizing the scintillation spectrum.
 *
 * \c yield is the characteristic light yield of the material.
 * \c resolution_scale scales the standard deviation of the distribution of the
 * number of photons generated.
 */
struct ScintillationSpectrum
{
    real_type yield{};
    real_type resolution_scale{};
    ItemRange<ScintillationComponent> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield > 0 && resolution_scale >= 0 && !components.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scintillation data tabulated with the optical material id.
 */
template<Ownership W, MemSpace M>
struct ScintillationData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    Items<ScintillationComponent> components;
    OpticalMaterialItems<ScintillationSpectrum> spectra;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !components.empty() && !spectra.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        components = other.components;
        spectra = other.spectra;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
