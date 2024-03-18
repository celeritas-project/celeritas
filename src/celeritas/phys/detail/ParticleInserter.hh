//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/ParticleInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Units.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build particle params data.
 */
class ParticleInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<ParticleParamsData>;
    using Input = ParticleParams::ParticleInput;
    using Id = ParticleId;
    //!@}

  public:
    // Construct from host data to be built
    explicit inline ParticleInserter(Data* data);

    // Add a particle type
    Id operator()(Input const& inp);

  private:
    template<class T>
    using Builder = CollectionBuilder<T, MemSpace::host, ParticleId>;

    Builder<units::MevMass> mass_;
    Builder<units::ElementaryCharge> charge_;
    Builder<real_type> decay_constant_;
    Builder<MatterType> matter_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from host data to be built.
 */
ParticleInserter::ParticleInserter(Data* data)
    : mass_{&data->mass}
    , charge_{&data->charge}
    , decay_constant_{&data->decay_constant}
    , matter_{&data->matter}
{
}

//---------------------------------------------------------------------------//
/*!
 * Add a particle.
 */
auto ParticleInserter::operator()(Input const& inp) -> Id
{
    CELER_EXPECT(inp.mass >= zero_quantity());
    CELER_EXPECT(inp.decay_constant >= 0);

    auto result = mass_.push_back(inp.mass);
    charge_.push_back(inp.charge);
    decay_constant_.push_back(inp.decay_constant);
    matter_.push_back(inp.pdg_code.get() < 0 ? MatterType::antiparticle
                                             : MatterType::particle);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
