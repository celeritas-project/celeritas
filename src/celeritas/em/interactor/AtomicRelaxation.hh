//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/AtomicRelaxation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/MiniStack.hh"
#include "corecel/cont/Span.hh"
#include "corecel/random/distribution/Selector.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simulate particle emission from atomic deexcitation.
 *
 * The EADL radiative and non-radiative transition data is used to simulate the
 * emission of fluorescence photons and (optionally) Auger electrons given an
 * initial shell vacancy created by a primary process.
 */
class AtomicRelaxation
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

    struct result_type
    {
        size_type count{};  //!< Number of secondaries created
        Energy energy{};  //!< Sum of the energies of the secondaries
    };

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION AtomicRelaxation(AtomicRelaxParamsRef const& shared,
                                           CutoffView const& cutoffs,
                                           ElementId el_id,
                                           SubshellId shell_id,
                                           Span<Secondary> secondaries,
                                           Span<SubshellId> vacancies);

    // Simulate atomic relaxation with an initial vacancy in the given shell ID
    template<class Engine>
    inline CELER_FUNCTION result_type operator()(Engine& rng);

  private:
    // Shared EADL atomic relaxation data
    AtomicRelaxParamsRef const& shared_;
    // Photon production threshold [MeV]
    Energy gamma_cutoff_;
    // Electron production threshold [MeV]
    Energy electron_cutoff_;
    // Index in MaterialParams elements
    ElementId el_id_;
    // Shell ID of the initial vacancy
    SubshellId shell_id_;
    // Fluorescence photons and Auger electrons
    Span<Secondary> secondaries_;
    // Storage for stack of unprocessed subshell vacancies
    Span<SubshellId> vacancies_;
    // Angular distribution of secondaries
    IsotropicDistribution<real_type> sample_direction_;

  private:
    using TransitionId = OpaqueId<AtomicRelaxTransition>;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The secondaries must have enough storage allocated for particles produced in
 * atomic relaxation and the vacancies must have enough storage allocated for
 * the stack of subshell IDs: this should be handled in code *before*
 * construction.
 *
 * The precondition of the element having relaxation data is satisfied by the
 * AtomicRelaxationHelper -- it is only "true" if a distribution can be
 * emitted.
 */
CELER_FUNCTION
AtomicRelaxation::AtomicRelaxation(AtomicRelaxParamsRef const& shared,
                                   CutoffView const& cutoffs,
                                   ElementId el_id,
                                   SubshellId shell_id,
                                   Span<Secondary> secondaries,
                                   Span<SubshellId> vacancies)
    : shared_(shared)
    , gamma_cutoff_(cutoffs.energy(shared_.ids.gamma))
    , electron_cutoff_(cutoffs.energy(shared_.ids.electron))
    , el_id_(el_id)
    , shell_id_(shell_id)
    , secondaries_(secondaries)
    , vacancies_(vacancies)
{
    CELER_EXPECT(shared_ && el_id_ < shared_.elements.size()
                 && shared_.elements[el_id_]);
    CELER_EXPECT(shell_id);
}

//---------------------------------------------------------------------------//
/*!
 * Simulate atomic relaxation with an initial vacancy in the given shell ID
 */
template<class Engine>
CELER_FUNCTION AtomicRelaxation::result_type
AtomicRelaxation::operator()(Engine& rng)
{
    AtomicRelaxElement const& el = shared_.elements[el_id_];
    auto const& shells = shared_.shells[el.shells];
    MiniStack<SubshellId> vacancies(vacancies_);

    // Push the vacancy created by the primary process onto a stack.
    vacancies.push(shell_id_);

    // Total number of secondaries
    size_type count = 0;
    real_type sum_energy = 0;

    // Generate the shower of photons and electrons produced by radiative and
    // non-radiative transitions
    while (!vacancies.empty())
    {
        // Pop the vacancy off the stack and check if it has transition data
        SubshellId vacancy_id = vacancies.pop();
        if (vacancy_id.get() >= shells.size())
            continue;

        // Sample a transition using shell probabilities
        AtomicRelaxSubshell const& shell = shells[vacancy_id.get()];
        auto transitions = shared_.transitions[shell.transitions];
        TransitionId const trans_id = make_unnormalized_selector(
            [&transitions](TransitionId i) {
                CELER_ASSERT(i < transitions.size());
                return transitions[i.unchecked_get()].probability;
            },
            id_cast<TransitionId>(transitions.size()),
            real_type{1})(rng);

        if (trans_id == id_cast<TransitionId>(transitions.size()))
        {
            // No transition sampled (probability: 1 - sum of transitions)
            continue;
        }

        // Push the new vacancies onto the stack and create the secondary
        auto const& transition = transitions[trans_id.get()];
        vacancies.push(transition.initial_shell);
        if (transition.auger_shell)
        {
            vacancies.push(transition.auger_shell);

            if (transition.energy >= electron_cutoff_)
            {
                // Sampled a non-radiative transition: create an Auger electron
                CELER_ASSERT(count < secondaries_.size());
                Secondary& secondary = secondaries_[count++];
                secondary.direction = sample_direction_(rng);
                secondary.energy = transition.energy;
                secondary.particle_id = shared_.ids.electron;

                // Accumulate the energy carried away by secondaries
                sum_energy += value_as<Energy>(transition.energy);
            }
        }
        else if (transition.energy >= gamma_cutoff_)
        {
            // Sampled a radiative transition: create a fluorescence photon
            CELER_ASSERT(count < secondaries_.size());
            Secondary& secondary = secondaries_[count++];
            secondary.direction = sample_direction_(rng);
            secondary.energy = transition.energy;
            secondary.particle_id = shared_.ids.gamma;

            // Accumulate the energy carried away by secondaries
            sum_energy += value_as<Energy>(transition.energy);
        }
    }

    result_type result;
    result.count = count;
    result.energy = Energy{sum_energy};
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
