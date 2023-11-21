//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//

enum class MatterType : char
{
    particle,
    antiparticle
};

//---------------------------------------------------------------------------//
/*!
 * Access particle definitions on the device.
 *
 * Fundamental (static) properties of a particle type. Physical state of a
 * particle (kinetic energy, ...) is part of a ParticleState. This view is
 * created from \c ParticleParams. The size of the \c defs data member is the
 * number of particle types (accessed by \c ParticleId).
 *
 * \sa ParticleParams (owns the pointed-to data)
 * \sa ParticleTrackView (uses the pointed-to data in a kernel)
 */
template<Ownership W, MemSpace M>
struct ParticleParamsData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::Collection<T, W, M, ParticleId>;

    //// DATA ////

    Items<units::MevMass> mass;  //!< Rest mass [MeV / c^2]
    Items<units::ElementaryCharge> charge;  //!< Charge in units of [e]
    Items<real_type> decay_constant;  //!< Decay constant [1/s]
    Items<MatterType> matter;  //!< Antiparticle (negative PDG
                               //!< number)

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !mass.empty() && !charge.empty() && !decay_constant.empty()
               && !matter.empty();
    }

    //! Params size
    CELER_FUNCTION typename Items<real_type>::size_type size() const
    {
        return decay_constant.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParticleParamsData& operator=(ParticleParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        mass = other.mass;
        charge = other.charge;
        decay_constant = other.decay_constant;
        matter = other.matter;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Physical (dynamic) state of a particle track.
 *
 * The "physical state" is just about what differentiates this particle from
 * another (type, energy, polarization, ...) in the lab's inertial reference
 * frame. It does not include information about the particle's direction or
 * position, nor about path lengths or collisions.
 *
 * The energy is with respect to the lab frame. The particle state is
 * immutable: collisions and other interactions should return changes to the
 * particle state.
 */

struct ParticleTrackInitializer
{
    ParticleId particle_id;  //!< Type of particle (electron, gamma, ...)
    units::MevEnergy energy;  //!< Kinetic energy [MeV]

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return particle_id && energy > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data storage/access for particle properties.
 *
 * \sa ParticleTrackView (uses the pointed-to data in a kernel)
 */
template<Ownership W, MemSpace M>
struct ParticleStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<ParticleId> particle_id;  //!< Type of particle (electron, gamma,
                                    //!< ...)
    Items<real_type> particle_energy;  //!< Kinetic energy [MeV]

    //// METHODS ////

    //! Whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !particle_id.empty() && !particle_energy.empty();
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return particle_id.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParticleStateData& operator=(ParticleStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        particle_id = other.particle_id;
        particle_energy = other.particle_energy;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize particle states in host code.
 */
template<MemSpace M>
inline void resize(ParticleStateData<Ownership::value, M>* data,
                   HostCRef<ParticleParamsData> const&,
                   size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&data->particle_id, size);
    resize(&data->particle_energy, size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
