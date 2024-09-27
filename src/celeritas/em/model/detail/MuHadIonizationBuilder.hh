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
 * This small helper class constructs and validates the data for the muon and
 * hadron ionization models.
 */
class MuHadIonizationBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using SetApplicability = std::set<Applicability>;
    //!@}

  public:
    // Construct with shared particle data and model label
    inline MuHadIonizationBuilder(ParticleParams const& particles,
                                  std::string_view label);

    // Construct model data from applicability
    inline MuHadIonizationData operator()(SetApplicability const&) const;

  private:
    ParticleParams const& particles_;
    std::string_view label_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared particle data and model label.
 */
MuHadIonizationBuilder::MuHadIonizationBuilder(ParticleParams const& particles,
                                               std::string_view label)
    : particles_(particles), label_(label)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct model data from applicability.
 */
MuHadIonizationData
MuHadIonizationBuilder::operator()(SetApplicability const& applicability) const
{
    CELER_EXPECT(!applicability.empty());

    MuHadIonizationData data;

    for (auto const& applic : applicability)
    {
        CELER_VALIDATE(
            applic,
            << "invalid applicability with particle `"
            << (applic.particle ? particles_.id_to_label(applic.particle) : "")
            << "' and energy limits (" << value_as<Energy>(applic.lower)
            << ", " << value_as<Energy>(applic.upper) << ") [MeV] for model '"
            << label_ << "'");
    }

    data.electron = particles_.find(pdg::electron());
    CELER_VALIDATE(data.electron,
                   << "missing electron (required for model '" << label_
                   << "')");

    data.electron_mass = particles_.get(data.electron).mass();

    CELER_ENSURE(data);
    return data;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
