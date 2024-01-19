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

#include "OpticalPropertyData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Material dependent scintillation spectrum.
 *
 * Support up to 3 components: [fast, medium, slow]
 */
struct ScintillationSpectrum
{
    static inline constexpr size_type size = 3;
    using Real3 = Array<real_type, size>;

    Real3 yield_prob{};  //!< Probability of the yield
    Real3 lambda_mean{};  //!< Mean wavelength
    Real3 lambda_sigma{};  //!< Standard dev. of wavelength
    Real3 rise_time{};  //!< Rise time
    Real3 fall_time{};  //!< Decay time
};

//---------------------------------------------------------------------------//
/*!
 *  Scintillation data tabulated with the optical material id.
 */
template<Ownership W, MemSpace M>
struct ScintillationData
{
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    OpticalMaterialItems<ScintillationSpectrum> spectrum;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !spectrum.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        spectrum = other.spectrum;
        return *this;
    }
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
