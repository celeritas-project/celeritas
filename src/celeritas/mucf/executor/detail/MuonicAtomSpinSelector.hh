//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/MuonicAtomSpinSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "celeritas/Types.hh"
#include "celeritas/mucf/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select muonic atom spin, in units of \f$ \frac{\hbar}{2} \f$.
 *
 * Sampling is based on spin population probabilities from [ \todo
 * https://doi.org/10.1038/s41598-022-09487-0 ], which are:
 * - Muonic deuterium: 2/3 probability for spin 3/2; 1/3 for spin 1/2
 * - Muonic tritium: 3/4 probability for spin 1; 1/4 for spin 0
 */
class MuonicAtomSpinSelector
{
  public:
    // Construct with muonic atom type
    inline CELER_FUNCTION MuonicAtomSpinSelector(MucfMuonicAtom atom);

    // Sample and return a spin value in units of hbar / 2
    template<class Engine>
    inline CELER_FUNCTION size_type operator()(Engine& rng);

  private:
    MucfMuonicAtom atom_;

    // Muonic deuterium spin probabilities: 2/3 for spin 3/2; 1/3 for spin 1/2
    constexpr CELER_FUNCTION real_type deuterium_spin_probability()
    {
        return real_type{2} / real_type{3};
    }

    // Muonic tritium spin probabilities: 3/4 for spin 1; 1/4 for spin 0
    constexpr CELER_FUNCTION real_type tritium_spin_probability()
    {
        return real_type{0.75};
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with muonic atom.
 */
CELER_FUNCTION
MuonicAtomSpinSelector::MuonicAtomSpinSelector(MucfMuonicAtom atom)
    : atom_(atom)
{
    CELER_EXPECT(atom_ < MucfMuonicAtom::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Select a muonic atom spin, in units of \f$ \hbar / 2 \f$.
 */
template<class Engine>
CELER_FUNCTION size_type MuonicAtomSpinSelector::operator()(Engine& rng)
{
    switch (atom_)
    {
        case MucfMuonicAtom::deuterium: {
            if (generate_canonical(rng) < deuterium_spin_probability())
            {
                return 3;  // Spin 3/2
            }
            else
            {
                return 1;  // Spin 1/2
            }
        }
        case MucfMuonicAtom::tritium: {
            if (generate_canonical(rng) < tritium_spin_probability())
            {
                return 2;  // Spin 1
            }
            else
            {
                return 0;  // Spin 0
            }
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
