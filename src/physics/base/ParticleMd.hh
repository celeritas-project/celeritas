//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleMd.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include "base/OpaqueId.hh"

namespace celeritas
{
struct ParticleMd;
using PDGNumber = OpaqueId<ParticleMd, int>;

//---------------------------------------------------------------------------//
/*!
 * Host-only metadata for a particle type.
 *
 * The PDG Monte Carlo number is a unique "standard model" identifier for a
 * particle. See "Monte Carlo Particle Numbering Scheme" in the "Review of
 * Particle Physics":
 * http://pdg.lbl.gov/2019/reviews/rpp2019-rev-monte-carlo-numbering.pdf
 *
 */
struct ParticleMd
{
    std::string name;     // Particle name
    PDGNumber   pdg_code; // See "Review of Particle Physics"
};

//---------------------------------------------------------------------------//
/*!
 * \namespace pdg
 *
 * Unique standard model particle identifiers by the Particle Data Group.
 *
 * These should be extended as needed.
 */
namespace pdg
{
//---------------------------------------------------------------------------//
//@{
//! Particle Data Group Monte Carlo number codes.
#define CELER_DEFINE_PDGNUMBER(NAME, VALUE) \
    inline constexpr PDGNumber NAME() { return PDGNumber{VALUE}; }

CELER_DEFINE_PDGNUMBER(electron, 11)
CELER_DEFINE_PDGNUMBER(gamma, 22)
CELER_DEFINE_PDGNUMBER(neutron, 2112)

#undef CELER_DEFINE_PDGNUMBER
//@}
//---------------------------------------------------------------------------//
} // namespace pdg
} // namespace celeritas

//---------------------------------------------------------------------------//
