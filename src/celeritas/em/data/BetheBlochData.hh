//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/BetheBlochData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data for the Bethe-Bloch ionization model.
 */
template<Ownership W, MemSpace M>
struct BetheBlochData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Particle IDs
    Items<ParticleId> particles;  //!< Model-dependent incident particles
    ParticleId electron;  //!< Secondary particle ID

    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //// METHODS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !particles.empty() && electron
               && electron_mass > zero_quantity();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    BetheBlochData& operator=(BetheBlochData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        particles = other.particles;
        electron = other.electron;
        electron_mass = other.electron_mass;
        return *this;
    }

    //! Whether the model is applicable to the given particle
    CELER_FUNCTION bool applies(ParticleId particle) const
    {
        return celeritas::any_of(
            particles.data().get(),
            particles.data().get() + particles.size(),
            [particle](ParticleId const& pid) { return pid == particle; });
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
