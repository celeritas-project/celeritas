//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/detail/CascadeParticle.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/FourVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Particle data to track the intra-nuclear cascade interactions.
 */
struct CascadeParticle
{
    //! Types of cascade particles in intra-nuclear interactions
    enum class ParticleType
    {
        unknown,
        proton,
        neutron,
        gamma
    };

    ParticleType type{ParticleType::unknown};  //!< Particle type
    units::MevMass mass;  // !< Particle mass
    FourVector four_vec;  //!< Four momentum in natural MevMomentum and
                          //!< MevEnergy units
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
