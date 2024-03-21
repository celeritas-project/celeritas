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
 * \c yield is the characteristic light yield of the material.
 * \c resolution_scale scales the standard deviation of the distribution of the
 * number of photons generated.
 * \c components stores the fast/slow/etc scintillation components for this
 * material.
 */
struct MaterialScintillationSpectrum
{
    real_type yield{};
    ItemRange<ScintillationComponent> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield > 0 && !components.empty();
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
 * - \c matid_to_optmatid returns an \c OpticalMaterialId given a
 *   \c MaterialId
 * - \c pid_to_scintpid returns a \c ScintillationParticleId given a
 *   \c ParticleId .
 * - \c resolution_scale is indexed by \c OpticalMaterialId .
 * - \c materials stores material-only scintillation data. Indexed by
 *   \c OpticalMaterialId
 * - \c particles stores scintillation data for each particle type for each
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
    using MaterialItems
        = Collection<MaterialScintillationSpectrum, W, M, OpticalMaterialId>;
    using ParticleItems
        = Collection<ParticleScintillationSpectrum, W, M, ParticleScintSpectrumId>;

    //// MEMBER DATA ////

    //! Index between OpticalMaterialId and MaterialId
    Collection<OpticalMaterialId, W, M, MaterialId> matid_to_optmatid;
    //! Index between ScintillationParticleId and ParticleId
    Collection<ScintillationParticleId, W, M, ParticleId> pid_to_scintpid;

    //! Resolution scale for each material
    Collection<real_type, W, M, OpticalMaterialId> resolution_scale;

    //! Material-only scintillation spectrum data
    MaterialItems materials;  //!< [OpticalMaterialId]

    //! Particle and material scintillation spectrum data
    ParticleItems particles;  //!< [ParticleScintSpectrumId]
    //! Backend storage for ParticleScintillationSpectrum::yield_vector
    Items<real_type> reals;

    //! Components for either material or particle items
    Items<ScintillationComponent> components;

    //! Cache number of scintillation particles; Used by this->spectrum_index
    size_type num_scint_particles{};

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !matid_to_optmatid.empty() && !pid_to_scintpid.empty()
               && (!materials.empty() || !particles.empty())
               && num_scint_particles > 0;
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
        CELER_EXPECT(pid < num_scint_particles && mat_id < materials.size());
        return ParticleScintSpectrumId{materials.size() * pid.get()
                                       + mat_id.get()};
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        matid_to_optmatid = other.matid_to_optmatid;
        pid_to_scintpid = other.pid_to_scintpid;
        resolution_scale = other.resolution_scale;
        materials = other.materials;
        particles = other.particles;
        reals = other.reals;
        components = other.components;
        num_scint_particles = other.num_scint_particles;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas