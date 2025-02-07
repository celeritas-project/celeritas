//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    units::MevMass mass;  //!< Particle mass
    FourVector four_vec;  //!< Four momentum in natural units
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
