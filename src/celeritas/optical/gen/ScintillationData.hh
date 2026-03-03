//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/ScintillationData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Types.hh"

#include "../Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Parameterized scintillation properties.
 *
 * This component represents one type of scintillation emissions, such as
 * prompt/fast, intermediate, or slow. It can be specific to a material or
 * depend on the incident particle type.
 *
 * \todo Could refactor as a distribution sampler: gaussian vs grid
 */
struct ScintRecord
{
    real_type lambda_mean{};  //!< Mean wavelength
    real_type lambda_sigma{};  //!< Standard deviation of wavelength
    real_type rise_time{};  //!< Rise time
    real_type fall_time{};  //!< Decay time
    ItemId<NonuniformGridRecord> energy_cdf;

    //! Whether this represents a normal distribution
    CELER_FUNCTION bool is_normal_distribution() const { return !energy_cdf; }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        bool const has_gauss = (lambda_mean > 0 && lambda_sigma > 0);
        bool const has_grid = static_cast<bool>(energy_cdf);
        return (has_gauss || has_grid) && rise_time >= 0 && fall_time > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Material-dependent scintillation spectrum.
 *
 * - \c yield_per_energy is the characteristic light yield of the material in
 *   [1/MeV] units. The total light yield per step is the characteristic light
 *   yield multiplied by the energy deposition, which results in a (unitless)
 *   number of photons.
 * - \c yield_pdf is the probability of choosing from a given component.
 * - \c components stores the different scintillation components
 *   (fast/slow/etc) for this material.
 */
struct MatScintSpectrum
{
    real_type yield_per_energy{};  //!< [1/MeV]
    ItemRange<real_type> yield_pdf;
    ItemRange<ScintRecord> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield_per_energy > 0 && !yield_pdf.empty()
               && yield_pdf.size() == components.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Particle- and material-dependent scintillation spectrum.
 *
 * - \c yield_vector is the characteristic light yield for different energies.
 * - \c yield_pdf is the probability of choosing from a given component.
 * - \c components stores the fast/slow/etc scintillation components for this
 * particle type.
 */
struct ParScintSpectrum
{
    NonuniformGridRecord yield_per_energy;  //! [MeV] -> [1/MeV]
    ItemRange<real_type> yield_pdf;
    ItemRange<ScintRecord> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield_per_energy && !yield_pdf.empty()
               && yield_pdf.size() == components.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data characterizing the scintillation spectrum for all materials.
 *
 * - \c resolution_scale is indexed by \c OptMatId .
 * - \c materials stores particle-independent scintillation data.
 */
template<Ownership W, MemSpace M>
struct ScintillationData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OptMatItems = Collection<T, W, M, OptMatId>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;
    template<class T>
    using ParScintSpectrumItems = Collection<T, W, M, ParScintSpectrumId>;

    //// MEMBER DATA ////

    //! Resolution scale for each material [OptMatId]
    OptMatItems<real_type> resolution_scale;
    //! Material-dependent scintillation spectrum data [OptMatId]
    OptMatItems<MatScintSpectrum> materials;

    // Cumulative probability of emission as a function of energy [MeV]
    Items<NonuniformGridRecord> energy_cdfs;

    //! Backend storage for real values
    Items<real_type> reals;
    //! Backend storage for scintillation components
    Items<ScintRecord> scint_records;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !resolution_scale.empty() && !materials.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        resolution_scale = other.resolution_scale;
        materials = other.materials;
        energy_cdfs = other.energy_cdfs;
        reals = other.reals;
        scint_records = other.scint_records;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
