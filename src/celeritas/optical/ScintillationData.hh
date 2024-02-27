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
//! Types
using ScintillationParticleId = OpaqueId<struct ScintillationParticle_>;
using ScintillationSpectrumId = OpaqueId<struct ScintillationSpectrum>;

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
 * Scintillation data tabulated with the optical particle id.
 */
template<Ownership W, MemSpace M>
struct ScintillationData
{
    template<class T>
    using Items = Collection<T, W, M>;

    //// MEMBER DATA ////

    // Grid: [scintillation id][optical material id]
    Collection<ScintillationParticleId, W, M, ParticleId> scint_particle_id;
    Items<ScintillationSpectrum> spectra;
    Items<ScintillationComponent> components;

    //! Number of optical materials
    size_type num_materials{};

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !particle_spectra.empty() && !spectra.empty()
               && !components.empty() && num_materials > 0;
    }

    ScintillationSpectrumId
    spectrum_index(ScintillationParticleId pid, OpticalMaterialId mat_id)
    {
        CELER_EXPECT(mat_id < num_materials && pid < scint_particle_id.size());
        return num_materials * pid.get() + mat_id.get();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scint_particle_id = other.scint_particle_id;
        spectra = other.spectra;
        components = other.components;
        num_materials = other.num_materials;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas