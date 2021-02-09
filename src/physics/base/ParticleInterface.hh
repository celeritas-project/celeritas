//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Pie.hh"
#include "base/Macros.hh"
#include "Types.hh"
#include "Units.hh"

#ifndef __CUDA_ARCH__
#    include "base/PieBuilder.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Fundamental (static) properties of a particle type.
 *
 * These should only be fundamental physical properties. Setting particles is
 * done through the ParticleParams. Physical state of a particle
 * (kinetic energy, ...) is part of a ParticleState.
 *
 * Particle definitions are accessed via the ParticleParams: using PDGs
 * to look up particle IDs, etc.
 */
struct ParticleDef
{
    units::MevMass          mass;           //!< Rest mass [MeV / c^2]
    units::ElementaryCharge charge;         //!< Charge in units of [e]
    real_type               decay_constant; //!< Decay constant [1/s]

    //! Value of decay_constant for a stable particle
    static CELER_CONSTEXPR_FUNCTION real_type stable_decay_constant()
    {
        return 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Access particle definitions on the device.
 *
 * This view is created from \c ParticleParams. The size of the \c defs data
 * member is the number of particle types (accessed by \c ParticleId).
 *
 * \sa ParticleParams (owns the pointed-to data)
 * \sa ParticleTrackView (uses the pointed-to data in a kernel)
 */
template<Ownership W, MemSpace M>
struct ParticleParamsData
{
    template<class T>
    using Data = celeritas::Pie<T, W, M>;

    Data<ParticleDef> particles;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !particles.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParticleParamsData& operator=(const ParticleParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        particles = other.particles;
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
struct ParticleTrackState
{
    ParticleId       particle_id; //!< Type of particle (electron, gamma, ...)
    units::MevEnergy energy;      //!< Kinetic energy [MeV]
};

//---------------------------------------------------------------------------//
/*!
 * View to the dynamic states of multiple physical particles.
 *
 * The size of the view will be the size of the vector of tracks. Each particle
 * track state corresponds to the thread ID (\c ThreadId).
 *
 * \sa ParticleTrackView (uses the pointed-to data in a kernel)
 */
template<Ownership W, MemSpace M>
struct ParticleStateData
{
    template<class T>
    using Data = celeritas::StatePie<T, W, M>;

    Data<ParticleTrackState> state;

    //! Whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION ThreadId::value_type size() const { return state.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParticleStateData& operator=(ParticleStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;
        return *this;
    }
};

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Resize particle states in host code.
 */
template<MemSpace M>
inline void
resize(ParticleStateData<Ownership::value, M>* data,
       const ParticleParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type size)
{
    CELER_EXPECT(size > 0);
    make_pie_builder(&data->state).resize(size);
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
