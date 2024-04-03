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
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Material dependent scintillation property.
 *
 * Components represent different scintillation emissions, such as
 * prompt/fast, intermediate, and slow. They can be material-only or depend on
 * the incident particle type.
 */
struct ScintillationComponent
{
    real_type yield_frac{};  //!< Fraction of total yield (yield/sum(yields))
    real_type lambda_mean{};  //!< Mean wavelength
    real_type lambda_sigma{};  //!< Standard dev. of wavelength
    real_type rise_time{};  //!< Rise time
    real_type fall_time{};  //!< Decay time

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield_frac > 0 && yield_frac <= 1 && lambda_mean > 0
               && lambda_sigma > 0 && rise_time >= 0 && fall_time > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data characterizing material-only scintillation spectrum information.
 *
 * - \c yield_per_energy is the characteristic light yield of the material in
 *   [1/MeV] units. The total light yield per step is then
 *   `yield_per_energy * energy_dep`, which results in a (unitless) number of
 *   photons.
 * - \c resolution_scale scales the standard deviation of the distribution of
 *   the number of photons generated.
 * - \c components stores the different scintillation components
 *   (fast/slow/etc) for this material.
 */
struct MaterialScintillationSpectrum
{
    real_type yield_per_energy{};  //!< [1/MeV]
    ItemRange<ScintillationComponent> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield_per_energy > 0 && !components.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data characterizing the scintillation spectrum for a given particle in a
 * given material.
 *
 * \c yield_vector is the characteristic light yield for different energies.
 * \c components stores the fast/slow/etc scintillation components for this
 * particle type.
 */
struct ParticleScintillationSpectrum
{
    GenericGridData yield_vector;
    ItemRange<ScintillationComponent> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(yield_vector);
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
 * - \c pid_to_scintpid returns a \c ScintillationParticleId given a
 *   \c ParticleId .
 * - \c resolution_scale is indexed by \c OpticalMaterialId .
 * - \c materials stores material-only scintillation data. Indexed by
 *   \c OpticalMaterialId
 * - \c particles stores scintillation spectrum for each particle type for each
 *   material, being a grid of size `num_particles * num_materials`. Therefore
 *   it is indexed by \c ParticleScintSpectrumId , which combines
 *   \c ScintillationParticleId and \c OpticalMaterialId . Use the
 *   \c spectrum_index() function to retrieve the correct index.
 */
template<Ownership W, MemSpace M>
struct ScintillationData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;
    using ParticleScintillationSpectra
        = Collection<ParticleScintillationSpectrum, W, M, ParticleScintSpectrumId>;

    //// MEMBER DATA ////

    //! Resolution scale for each material [OpticalMaterialid]
    OpticalMaterialItems<real_type> resolution_scale;

    //! Material-only scintillation spectrum data [OpticalMaterialid]
    OpticalMaterialItems<MaterialScintillationSpectrum> materials;

    //! Index between ScintillationParticleId and ParticleId
    Collection<ScintillationParticleId, W, M, ParticleId> pid_to_scintpid;
    //! Cache number of scintillation particles; Used by this->spectrum_index
    size_type num_scint_particles{};
    //! Particle/material scintillation spectrum data [ParticleScintSpectrumId]
    ParticleScintillationSpectra particles;
    //! Backend storage for ParticleScintillationSpectrum::yield_vector
    Items<real_type> reals;

    //! Components for either material or particle items
    Items<ScintillationComponent> components;

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
    ParticleScintSpectrumId
    spectrum_index(ScintillationParticleId pid, OpticalMaterialId mat_id) const
    {
        // Resolution scale exists independent of material-only data and it's
        // indexed by optical material id
        CELER_EXPECT(pid < num_scint_particles
                     && mat_id < resolution_scale.size());
        return ParticleScintSpectrumId{resolution_scale.size() * pid.get()
                                       + mat_id.get()};
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
        reals = other.reals;
        components = other.components;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas