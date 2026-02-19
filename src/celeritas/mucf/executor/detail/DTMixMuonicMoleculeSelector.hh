//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DTMixMuonicMoleculeSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "celeritas/Types.hh"
#include "celeritas/mucf/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a muonic molecule by calculating the interaction lengths of the
 * possible molecule formations.
 *
 * This is the equivalent of Geant4's
 * \c G4VRestProcess::AtRestGetPhysicalInteractionLength
 */
class DTMixMuonicMoleculeSelector
{
  public:
    //!@{
    //! \name Type aliases
    using CycleRatesArray = EnumArray<MucfMuonicMolecule, Array<real_type, 2>>;
    using MaterialFractionsArray = EnumArray<MucfIsotope, real_type>;
    using HalfSpinInt = units::HalfSpinInt;
    //!@}

    // Result of molecule and cycle time selection
    struct Result
    {
        MucfMuonicMolecule molecule;
        real_type cycle_time;  // [s]
    };

    //! Construct with muonic atom and material information
    inline CELER_FUNCTION
    DTMixMuonicMoleculeSelector(MucfMuonicAtom atom,
                                HalfSpinInt spin,
                                MaterialFractionsArray material_fractions,
                                CycleRatesArray cycle_rates);

    // Select muonic molecule
    template<class Engine>
    inline CELER_FUNCTION Result operator()(Engine&);

  private:
    MucfMuonicAtom atom_;
    HalfSpinInt spin_;
    MaterialFractionsArray material_fractions_;
    CycleRatesArray cycle_rates_;

    inline CELER_FUNCTION size_type spin_state_index(HalfSpinInt spin) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with args.
 *
 * \todo Update documentation
 */
CELER_FUNCTION
DTMixMuonicMoleculeSelector::DTMixMuonicMoleculeSelector(
    MucfMuonicAtom atom,
    HalfSpinInt spin,
    MaterialFractionsArray material_fractions,
    CycleRatesArray cycle_rates)
    : atom_(atom)
    , spin_(spin)
    , material_fractions_(material_fractions)
    , cycle_rates_(cycle_rates)
{
    CELER_EXPECT(atom < MucfMuonicAtom::size_);
    CELER_EXPECT(material_fractions_[MucfIsotope::deuterium]
                     + material_fractions_[MucfIsotope::tritium]
                 > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Return a muonic molecule by selecting the shortest sampled molecule
 * interaction time (\f$ T = -\ln(r) \times \tau_\text{cycle} \f$).
 */
template<class Engine>
CELER_FUNCTION DTMixMuonicMoleculeSelector::Result
DTMixMuonicMoleculeSelector::operator()(Engine& rng)
{
    using MMM = MucfMuonicMolecule;

    real_type shortest_time = std::numeric_limits<real_type>::max();
    MucfMuonicMolecule molecule{MMM::size_};
    for (auto mol : range(MMM::size_))
    {
        auto const& cycle_rate
            = cycle_rates_[mol][this->spin_state_index(spin_)];

        auto sampled_time = ExponentialDistribution<real_type>(cycle_rate)(rng);
        if (sampled_time < shortest_time)
        {
            shortest_time = sampled_time;
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
CELER_FUNCTION size_type
DTMixMuonicMoleculeSelector::spin_state_index(HalfSpinInt spin) const
{
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
}  // namespace detail
}  // namespace celeritas
