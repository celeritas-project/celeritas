//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/MuonicMoleculeSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
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
 * (\f$ T = -\ln(r) \times \tau_\text{cycle} \f$).
 *
 * \note The cycle rates cached in the model data are calculated based on the
 * material definition, and thus are propagated to the final cycle time sampled
 * here and used in the fusion process.
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
        MucfMuonicMolecule molecule;
        real_type cycle_time;  //!< in [s]
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
    HalfSpinInt spin_;
    CycleRatesArray cycle_rates_;

    // Return the correct array index for each reactive spin state
    inline CELER_FUNCTION size_type spin_state_index(MucfMuonicMolecule mol,
                                                     HalfSpinInt spin) const;

    // Sample the final cycle time for a given molecule
    template<class Engine>
    inline CELER_FUNCTION real_type sample_exp_time(MucfMuonicMolecule molecule,
                                                    Engine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with muonic atom and material information.
 */
CELER_FUNCTION
MuonicMoleculeSelector::MuonicMoleculeSelector(MucfMuonicAtom atom,
                                               HalfSpinInt spin,
                                               CycleRatesArray cycle_rates)
    : atom_(atom), spin_(spin), cycle_rates_(cycle_rates)
{
    CELER_EXPECT(atom < MucfMuonicAtom::size_);
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

    // Select final molecule and its respective cycle time
    CycleTimeArray sampled_times;
    real_type shortest_time = std::numeric_limits<real_type>::infinity();
    MMM molecule{MMM::size_};
    for (auto mol : range(MMM::size_))
    {
        sampled_times[mol] = this->sample_exp_time(mol, rng);
        if (sampled_times[mol] < shortest_time)
        {
            shortest_time = sampled_times[mol];
            molecule = mol;
        }
    }

    CELER_ENSURE(molecule < MMM::size_);
    return {molecule, shortest_time};
}

//---------------------------------------------------------------------------//
/*!
 * Return correct \c CycleRatesArray spin state index for each molecule.
 *
 * Given the limited number of states, this is manually hardcoded during data
 * construction by \c MucfMaterialInserter::calc_[dd|dt|tt]_cycle .
 */
CELER_FUNCTION size_type MuonicMoleculeSelector::spin_state_index(
    MucfMuonicMolecule mol, HalfSpinInt spin) const
{
    // The array index is wonky and I have to fix that...
    switch (atom_)
    {
        case MucfMuonicAtom::deuterium: {
            // Muonic deuterium has spin states 1/2 and 3/2, which correspond
            // to indices 0 and 1, respectively
            CELER_EXPECT(spin == HalfSpinInt{1} || spin == HalfSpinInt{3});
            return (spin == HalfSpinInt{1}) ? 0 : 1;
        }

        case MucfMuonicAtom::tritium: {
            // Muonic tritium has spin states 0 and 1, which correspond to
            // indices 0 and 1, respectively
            CELER_EXPECT(spin == HalfSpinInt{0} || spin == HalfSpinInt{1});
            return (spin == HalfSpinInt{0}) ? 0 : 1;
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample the final exponential distribution for a given cycle rate using
 * \f$ T = -\ln(r) \times \tau_\text{cycle} \f$ . If the cached rate is zero,
 * the returned time is set to infinity, which effectively removes that
 * molecule as a possible selection.
 */
template<class Engine>
CELER_FUNCTION real_type MuonicMoleculeSelector::sample_exp_time(
    MucfMuonicMolecule molecule, Engine& rng)
{
    CELER_EXPECT(molecule < MucfMuonicMolecule::size_);
    auto const rate = cycle_rates_[molecule][this->spin_state_index(spin_)];

    // Return infinity when the cached rate is zero
    return (rate > 0) ? ExponentialDistribution<real_type>(rate)(rng)
                      : std::numeric_limits<real_type>::infinity();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
