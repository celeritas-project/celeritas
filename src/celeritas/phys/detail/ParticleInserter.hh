//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/ParticleInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Units.hh"

#include "../ParticleData.hh"
#include "../ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build particle parameters from user input.
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
    inline Id operator()(Input const& inp);

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
    CELER_VALIDATE(inp.mass >= zero_quantity(),
                   << "invalid particle mass " << inp.mass.value());
    CELER_VALIDATE(inp.decay_constant >= 0,
                   << "invalid particle decay constant " << inp.decay_constant);

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
