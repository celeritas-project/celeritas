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
#include "celeritas/grid/XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Material dependent scintillation property.
 *
 * Components represent different scintillation emissions, such as
 * prompt/fast, intermediate, and slow. It is also dependend on the incident
 * particle type.
 */
struct ScintillationComponent
{
    real_type yield_frac{};  //!< Ratio of the total yield (yield/sum(yields))
    real_type lambda_mean{};  //!< Mean wavelength
    real_type lambda_sigma{};  //!< Standard dev. of wavelength
    real_type rise_time{};  //!< Rise time
    real_type fall_time{};  //!< Decay time

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return yield_frac > 0 && lambda_mean > 0 && lambda_sigma > 0
               && rise_time >= 0 && fall_time > 0;
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
 * Data characterizing the scintillation spectrum for a given particle in a
 * given material.
 *
 * \c yield_vector is the characteristic light yield for different energies.
 * \c components stores the fast/slow/etc scintillation components for this
 * particle type.
 */
struct ParticleScintillationSpectrum
{
    XsGridData yield_vector;
    ItemRange<ScintillationComponent> components;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(yield_vector);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data characterizing the scintillation spectrum.
 *
 * \c materials stores material-only scintillation data indexed by
 * \c OpticalMaterialId .
 * \c particles stores scintillation data for each particle type available and
 * is indexed by \c ScintillationParticleId and \c OpticalMaterialId .
 */
template<Ownership W, MemSpace M>
struct ScintillationData
{
    template<class T>
    using Items = Collection<T, W, M>;
    using MaterialItems
        = Collection<MaterialScintillationSpectrum, W, M, OpticalMaterialId>;
    using ParticleItems
        = Collection<ParticleScintillationSpectrum, W, M, ScintillationSpectrumId>;

    //// MEMBER DATA ////

    //! Index between OpticalMaterialId and MaterialId
    Collection<MaterialId, W, M, OpticalMaterialId> optical_mat_id;
    //! Index between ScintillationParticleId and ParticleId
    Collection<ParticleId, W, M, ScintillationParticleId> scint_particle_id;

    //! Material-only scintillation spectrum data
    MaterialItems materials;  //!< [OpticalMaterialId]
    Items<ScintillationComponent> material_components;

    //! Particle and material scintillation spectrum data
    ParticleItems particles;  //!< [ScintillationSpectrumId]
    Items<ScintillationComponent> particle_components;

    real_type num_materials{};
    real_type num_particles{};

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !materials.empty() && num_materials > 0;
    }

    //! Retrieve spectrum index for a given optical particle and material ids
    ScintillationSpectrumId
    spectrum_index(ScintillationParticleId pid, OpticalMaterialId mat_id) const
    {
        CELER_EXPECT(mat_id < num_materials && pid < scint_particle_id.size());
        return ScintillationSpectrumId{num_materials * pid.get()
                                       + mat_id.get()};
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ScintillationData& operator=(ScintillationData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        optical_mat_id = other.optical_mat_id;
        scint_particle_id = other.scint_particle_id;
        materials = other.materials;
        particles = other.particles;
        num_materials = other.num_materials;
        num_particles = other.num_particles;
        return *this;
    }
};

//---------------------------------------------------------------------------//
//! Type aliases
using ScintDataDeviceRef = DeviceCRef<ScintillationData>;
using ScintDataHostRef = HostCRef<ScintillationData>;
using ScintDataNativeRef = NativeCRef<ScintillationData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas