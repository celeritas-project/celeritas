//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/detail/MuHadIonizationBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>
#include <string_view>

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuHadIonizationData.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct muon and hadron ionization data.
 *
 * This small helper class constructs the common data and model-dependent
 * incident particles for the muon and hadron ionization models.
 */
class MuHadIonizationBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<MuHadIonizationData>;
    using Energy = units::MevEnergy;
    using SetApplicability = std::set<Applicability>;
    //!@}

  public:
    // Construct with shared particle data and model description
    explicit inline MuHadIonizationBuilder(ParticleParams const& particles,
                                           std::string_view description);

    // Construct model data from applicability
    inline Data operator()(SetApplicability const&) const;

  private:
    ParticleParams const& particles_;
    std::string_view description_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared particle data and model description.
 */
MuHadIonizationBuilder::MuHadIonizationBuilder(ParticleParams const& particles,
                                               std::string_view description)
    : particles_(particles), description_(description)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct model data from applicability.
 */
auto MuHadIonizationBuilder::operator()(
    SetApplicability const& applicability) const -> Data
{
    CELER_EXPECT(!applicability.empty());

    Data data;

    auto particles = make_builder(&data.particles);
    particles.reserve(applicability.size());
    for (auto const& applic : applicability)
    {
        CELER_VALIDATE(applic,
                       << "invalid applicability with particle ID "
                       << applic.particle.unchecked_get()
                       << " and energy limits ("
                       << value_as<Energy>(applic.lower) << ", "
                       << value_as<Energy>(applic.upper) << ") [MeV] for "
                       << description_);
        particles.push_back(applic.particle);
    }

    data.electron = particles_.find(pdg::electron());
    CELER_VALIDATE(data.electron,
                   << "missing electron (required for " << description_ << ")");

    data.electron_mass = particles_.get(data.electron).mass();

    CELER_ENSURE(data);
    return data;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
