//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/MuonicMoleculeSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mucf/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a muonic molecule and its final cycle time.
 *
 * The muonic molecule is selected by sampling the shortest fusion cycle time
 * \f[
 * T = -\ln(r) \times \tau_\text{cycle},
 * \f]
 *
 * where \f$ \tau_\text{cycle} \f$ is the cycle time a given molecule + spin
 * calculated from material data.
 */
class MuonicMoleculeSelector
{
  public:
    //!@{
    //! \name Type aliases
    using CycleRatesArray = EnumArray<MucfMuonicMolecule, Array<real_type, 2>>;
    using HalfSpinInt = units::HalfSpinInt;
    //!@}

    // Result of molecule and cycle time selection
    struct Result
    {
        MucfMuonicMolecule molecule{MucfMuonicMolecule::size_};
        real_type cycle_time{numeric_limits<real_type>::max()};

        //! Check whether the data are assigned
        explicit CELER_FUNCTION operator bool() const
        {
            return molecule < MucfMuonicMolecule::size_ && cycle_time > 0
                   && cycle_time < numeric_limits<real_type>::max();
        }
    };

    //! Construct with muonic atom and material information
    inline CELER_FUNCTION MuonicMoleculeSelector(MucfMuonicAtom atom,
                                                 HalfSpinInt spin,
                                                 CycleRatesArray cycle_rates);

    // Select muonic molecule and sample its final cycle time
    template<class Engine>
    inline CELER_FUNCTION Result operator()(Engine& rng);

  private:
    MucfMuonicAtom atom_;
    size_type cycle_rate_index_;
    CycleRatesArray cycle_rates_;

    // Sample the final cycle time for a given molecule
    template<class Engine>
    inline CELER_FUNCTION real_type sample_exp_time(MucfMuonicMolecule molecule,
                                                    Engine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with muonic atom and cycle rate information.
 *
 * The correct cycle rate array index is determined at construction by the
 * combination of atom+spin. Given the limited number of states, the position
 * in the index is set in \c MucfMaterialInserter::calc_[dd|dt|tt]_cycle
 * manually.
 *
 * \note Current implementation has no safe method to access cycle rate data
 * for each spin state.
 */
CELER_FUNCTION
MuonicMoleculeSelector::MuonicMoleculeSelector(MucfMuonicAtom atom,
                                               HalfSpinInt spin,
                                               CycleRatesArray cycle_rates)
    : atom_(atom), cycle_rates_(cycle_rates)
{
    CELER_EXPECT(atom < MucfMuonicAtom::size_);

    // Check that the spin value is valid for the given atom type and set the
    // cycle rate array index accordingly
    switch (atom_)
    {
        case MucfMuonicAtom::deuterium: {
            CELER_EXPECT(spin == spin_one_half || spin == spin_three_halves);
            // DD fusion
            // F = 1/2 and F = 3/2 correspond to indices 0 and 1, respectively
            cycle_rate_index_ = (spin == spin_one_half) ? 0 : 1;
            break;
        }
        case MucfMuonicAtom::tritium: {
            CELER_EXPECT(spin == spin_zero || spin == spin_one
                         || spin == spin_one_half);
            // DT: F = 0 and F = 1 correspond to indices 0 and 1, respectively
            // TT: F = 1/2 corresponds to index 0
            cycle_rate_index_
                = (spin == spin_zero || spin == spin_one_half) ? 0 : 1;
            break;
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return a muonic molecule and its sampled cycle time.
 */
template<class Engine>
CELER_FUNCTION MuonicMoleculeSelector::Result
MuonicMoleculeSelector::operator()(Engine& rng)
{
    using MMM = MucfMuonicMolecule;
    using CycleTimeArray = EnumArray<MucfMuonicMolecule, real_type>;

    CycleTimeArray sampled_times;
    auto const inf = numeric_limits<real_type>::max();
    if (atom_ == MucfMuonicAtom::deuterium)
    {
        // DD fusion is only triggered by a muonic deuterium
        sampled_times[MMM::deuterium_deuterium]
            = this->sample_exp_time(MMM::deuterium_deuterium, rng);
        sampled_times[MMM::deuterium_tritium] = inf;
        sampled_times[MMM::tritium_tritium] = inf;
    }
    else
    {
        // DT and TT fusions are triggered by a muonic tritium
        sampled_times[MMM::deuterium_deuterium] = inf;
        sampled_times[MMM::deuterium_tritium]
            = this->sample_exp_time(MMM::deuterium_tritium, rng);
        sampled_times[MMM::tritium_tritium]
            = this->sample_exp_time(MMM::tritium_tritium, rng);
    }

    Result result;
    for (auto mol : range(MMM::size_))
    {
        if (sampled_times[mol] < result.cycle_time)
        {
            result.cycle_time = sampled_times[mol];
            result.molecule = mol;
        }
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the final exponential distribution for a given cycle rate using
 * \f$ T = -\ln(r) \times \tau_\text{cycle} \f$ . If the cached rate is
 * zero, the returned time is set to infinity, which effectively removes
 * that molecule as a possible selection.
 */
template<class Engine>
CELER_FUNCTION real_type MuonicMoleculeSelector::sample_exp_time(
    MucfMuonicMolecule molecule, Engine& rng)
{
    CELER_EXPECT(molecule < MucfMuonicMolecule::size_);
    auto const rate = cycle_rates_[molecule][cycle_rate_index_];

    // Return an infinite cycle time when the cached rate is zero
    return (rate > 0) ? ExponentialDistribution<real_type>(rate)(rng)
                      : numeric_limits<real_type>::max();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
