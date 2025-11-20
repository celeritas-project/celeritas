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
 */
struct ScintRecord
{
    real_type lambda_mean{};  //!< Mean wavelength
    real_type lambda_sigma{};  //!< Standard deviation of wavelength
    real_type rise_time{};  //!< Rise time
    real_type fall_time{};  //!< Decay time
    ItemId<NonuniformGridRecord> energy_cdf;
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
 * Data characterizing the scintillation spectrum for all particles and
 * materials.
 *
 * Sampling using material-only data or particle- and material-dependent data
 * are mutually exclusive. Therefore, either \c materials or \c particles are
 * loaded at the beginning of the simulation, but *never* both at the same
 * time. The \c scintillation_by_particle() function can be used to check that.
 *
 * - \c pid_to_scintpid maps a \c ParticleId to a \c ScintParticleId .
 * - \c resolution_scale is indexed by \c OptMatId .
 * - \c materials stores particle-independent scintillation data.
 * - \c particles stores the scintillation spectrum for each particle type and
 *   material. It has size \c num_particles * \c num_materials and is indexed
 *   by \c ParScintSpectrumId , which can be calculated from a \c OptMatId and
 *   \c ScintParticleId using the \c spectrum_index() helper method.
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

    //! Number of scintillation particles, used by this->spectrum_index
    size_type num_scint_particles{};

    //! Resolution scale for each material [OptMatId]
    OptMatItems<real_type> resolution_scale;
    //! Material-dependent scintillation spectrum data [OptMatId]
    OptMatItems<MatScintSpectrum> materials;

    // Cumulative probability of emission as a function of energy [MeV]
    Items<NonuniformGridRecord> energy_cdfs;
    //! Index between \c ScintParticleId and \c ParticleId
    ParticleItems<ScintParticleId> pid_to_scintpid;
    //! Particle/material scintillation spectrum data [ParScintSpectrumId]
    ParScintSpectrumItems<ParScintSpectrum> particles;

    //! Backend storage for real values
    Items<real_type> reals;
    //! Backend storage for scintillation components
    Items<ScintRecord> scint_records;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !resolution_scale.empty()
               && (materials.empty() != particles.empty())
               && (!pid_to_scintpid.empty() == !particles.empty())
               && (!pid_to_scintpid.empty() == (num_scint_particles > 0));
    }

    //! Whether sampling must happen by particle type
    CELER_FUNCTION bool scintillation_by_particle() const
    {
        return !particles.empty();
    }

    //! Retrieve spectrum index given optical particle and material ids
    ParScintSpectrumId spectrum_index(ScintParticleId pid, OptMatId mid) const
    {
        // Resolution scale exists independent of material-only data and it's
        // indexed by optical material id
        CELER_EXPECT(pid < num_scint_particles);
        CELER_EXPECT(mid < resolution_scale.size());
        return ParScintSpectrumId{resolution_scale.size() * pid.get()
                                  + mid.get()};
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        resolution_scale = other.resolution_scale;
        materials = other.materials;
        pid_to_scintpid = other.pid_to_scintpid;
        num_scint_particles = other.num_scint_particles;
        particles = other.particles;
        energy_cdfs = other.energy_cdfs;
        reals = other.reals;
        scint_records = other.scint_records;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
