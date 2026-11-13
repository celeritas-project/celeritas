//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/DecayPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>

#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Muon decay to an electron or positron.
 *
 * Muons decay as \f$ \mu^- \to e^- \bar{\nu}_e \nu_\mu \f$ and
 * \f$ \mu^+ \to e^+ \nu_e \bar{\nu}_\mu \f$ with a branching ratio of one.
 * The daughters are fixed by the parent, so only the decaying particles are
 * specified. Neutrinos are not produced.
 */
struct MuonDecay
{
    //! Particles that decay through this channel: \c mu- and/or \c mu+
    std::set<PDGNumber> parents;

    //! Whether muon decay is enabled
    explicit operator bool() const { return !parents.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Particle decay processes.
 *
 * Each supported decay channel is listed separately and knows which
 * particles it applies to and what it produces. Currently each particle
 * may decay through at most one channel. The decay rate comes from the
 * particle's lifetime (see \c inp::Particle ).
 */
struct DecayPhysics
{
    //! Muon decay
    MuonDecay muon;

    //! Whether any decay channel is enabled
    explicit operator bool() const { return static_cast<bool>(muon); }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
